import torch
import time
import queue
from typing import List, Dict, Any
from ..agents.llm_agent import LLMAgent

class ModelServer:
    """
    Central model server that processes inference requests from multiple workers
    """
    def __init__(
        self, 
        model, 
        tokenizer, 
        request_queue, 
        response_queues: Dict[int, Any],
        batch_size: int = 32,
        timeout: float = 0.01
    ):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure pad_token is set for batch generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = True

    def run(self):
        """Main server loop"""
        print(f"Model Server started. Batch size: {self.batch_size}")
        
        while self.running:
            batch_requests = []
            
            # 1. Collect batch
            start_wait = time.time()
            while len(batch_requests) < self.batch_size:
                try:
                    # If we have some items, use short timeout. If empty, wait longer.
                    current_timeout = self.timeout if batch_requests else 0.1
                    
                    req = self.request_queue.get(timeout=current_timeout)
                    
                    if req is None: # Poison pill
                        self.running = False
                        break
                        
                    batch_requests.append(req)
                    
                except queue.Empty:
                    # If we have collected some items, stop waiting and process them
                    if batch_requests:
                        break
                    # If empty, continue waiting (outer loop)
            
            if not self.running:
                break
                
            if batch_requests:
                self._process_batch(batch_requests)

    def _process_batch(self, requests: List[Dict[str, Any]]):
        """Process a batch of requests"""
        # Group by generation config (max_tokens, temperature)
        # For simplicity, we assume all requests use same config for now
        # or we just use the config from the first request
        
        prompts = [r['prompt'] for r in requests]
        worker_ids = [r['worker_id'] for r in requests]
        request_ids = [r['request_id'] for r in requests]
        
        max_tokens = requests[0].get('max_tokens', 256)
        temperature = requests[0].get('temperature', 0.7)
        
        # --- CRITICAL: Handle Left Padding ---
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
            
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Process results
        prompt_len = inputs.input_ids.shape[1]
        
        for i, (worker_id, req_id) in enumerate(zip(worker_ids, request_ids)):
            # Extract sequence
            gen_ids = outputs.sequences[i][prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # Extract scores
            # outputs.scores is tuple of (batch, vocab), slice for this sample
            sample_scores = tuple(score[i:i+1] for score in outputs.scores)
            
            # Compute log prob
            log_prob = LLMAgent.compute_log_prob_from_scores(sample_scores, gen_ids)
            
            # Prepare result
            result = {
                "request_id": req_id,
                "text": text,
                "log_prob": log_prob,
                "input_ids": inputs.input_ids[i].cpu(),
                "generated_ids": gen_ids.cpu()
            }
            
            # Send back to specific worker
            self.response_queues[worker_id].put(result)

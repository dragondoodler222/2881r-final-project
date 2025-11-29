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
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 128001
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 128001
        
        for i, (worker_id, req_id) in enumerate(zip(worker_ids, request_ids)):
            # Extract sequence
            gen_ids = outputs.sequences[i][prompt_len:]
            
            # Strip trailing padding/EOS tokens from generated_ids
            # These are added for batch padding but weren't actually sampled
            stripped_gen_ids = gen_ids.clone()
            # Find first EOS/pad token and truncate
            for end_pos in range(len(gen_ids)):
                if gen_ids[end_pos].item() == eos_token_id or gen_ids[end_pos].item() == pad_token_id:
                    # Include this token if it's EOS (but not subsequent padding)
                    if gen_ids[end_pos].item() == eos_token_id:
                        stripped_gen_ids = gen_ids[:end_pos+1]
                    else:
                        stripped_gen_ids = gen_ids[:end_pos]
                    break
            
            text = self.tokenizer.decode(stripped_gen_ids, skip_special_tokens=True)
            
            # Extract scores (only up to stripped length)
            # outputs.scores is tuple of (batch, vocab), slice for this sample
            sample_scores = tuple(score[i:i+1] for score in outputs.scores[:len(stripped_gen_ids)])
            
            # Compute log prob (passing token IDs for proper EOS/pad handling)
            # Use the static method we just added to LLMAgent
            log_prob = LLMAgent.compute_log_prob_from_scores(
                sample_scores, stripped_gen_ids, 
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id
            )
            
            # Prepare result - use stripped generated_ids
            result = {
                "request_id": req_id,
                "text": text,
                "log_prob": log_prob,
                "input_ids": inputs.input_ids[i].cpu(),
                "generated_ids": stripped_gen_ids.cpu()  # Store stripped version
            }
            
            # Send back to specific worker
            self.response_queues[worker_id].put(result)

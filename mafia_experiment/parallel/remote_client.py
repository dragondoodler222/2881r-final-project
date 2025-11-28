import uuid
import time
from typing import List, Dict, Any

class RemoteModelClient:
    """
    Client that sends generation requests to the ModelServer
    """
    def __init__(self, request_queue, response_queue, worker_id: int):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.worker_id = worker_id

    def generate_batch(self, prompts: List[str], max_tokens=256, temperature=0.7) -> List[Dict[str, Any]]:
        """
        Send batch of prompts to server and wait for results
        """
        # Create requests
        request_ids = []
        for prompt in prompts:
            req_id = str(uuid.uuid4())
            request_ids.append(req_id)
            
            req = {
                "worker_id": self.worker_id,
                "request_id": req_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            self.request_queue.put(req)
            
        # Wait for responses
        # Note: Responses might come out of order if server processes multiple batches
        # But since this client blocks until all its requests are done, we just need to collect N responses
        
        results_map = {}
        pending_count = len(prompts)
        
        while pending_count > 0:
            resp = self.response_queue.get()
            results_map[resp['request_id']] = resp
            pending_count -= 1
            
        # Return in original order
        ordered_results = []
        for req_id in request_ids:
            ordered_results.append(results_map[req_id])
            
        return ordered_results

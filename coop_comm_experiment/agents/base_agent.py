"""
Base Agent: Abstract base class for communication experiment agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import torch


@dataclass
class AgentResponse:
    """Represents a response from an agent"""
    agent_id: str
    response_type: str  # "message", "answer", "prediction"
    content: str
    log_prob: float = 0.0
    token_log_probs: Optional[List[float]] = None
    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    raw_text: str = ""
    confidence: float = 1.0


class BaseCommAgent(ABC):
    """
    Abstract base class for agents in the communication experiment.
    """
    
    def __init__(
        self,
        agent_id: str,
        model: Any = None,
        tokenizer: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 100
    ):
        self.agent_id = agent_id
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # For RL training
        self.last_log_prob: Optional[float] = None
        self.last_token_log_probs: Optional[List[float]] = None
        self.last_prompt: str = ""
        self.last_input_ids: Optional[torch.Tensor] = None
        self.last_generated_ids: Optional[torch.Tensor] = None
        self.last_raw_response: str = ""
    
    @abstractmethod
    def generate_message(
        self,
        private_info: str,
        question: str,
        received_messages: List[Dict[str, str]],
        message_round: int,
        is_cot_public: bool = False
    ) -> AgentResponse:
        """
        Generate a message to send to the other agent.
        
        Args:
            private_info: Agent's private information
            question: The shared question
            received_messages: Messages received from other agent
            message_round: Current round of message exchange
            is_cot_public: Whether chain-of-thought is visible
            
        Returns:
            AgentResponse with the message
        """
        pass
    
    @abstractmethod
    def generate_answer(
        self,
        private_info: str,
        question: str,
        all_messages: List[Dict[str, str]],
        is_cot_public: bool = False
    ) -> AgentResponse:
        """
        Generate final True/False answer.
        
        Args:
            private_info: Agent's private information
            question: The shared question
            all_messages: All messages exchanged
            is_cot_public: Whether chain-of-thought is visible
            
        Returns:
            AgentResponse with True/False answer
        """
        pass
    
    def _generate_with_log_prob(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float, torch.Tensor, torch.Tensor, List[float]]:
        """
        Generate text and compute log probability.
        
        Returns:
            Tuple of (generated_text, total_log_prob, input_ids, generated_ids, token_log_probs)
        """
        max_tokens = max_tokens or self.max_tokens
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        input_ids = inputs.input_ids.cpu().clone()
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract generated tokens
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_ids_cpu = generated_ids.cpu().clone()
        
        # Compute log probabilities
        log_prob_sum, token_log_probs = self._compute_log_probs(outputs.scores, generated_ids)
        
        return generated_text, log_prob_sum, input_ids, generated_ids_cpu, token_log_probs
    
    def _compute_log_probs(
        self,
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor
    ) -> Tuple[float, List[float]]:
        """Compute log probabilities from generation scores."""
        import torch.nn.functional as F
        
        log_prob_sum = 0.0
        token_log_probs = []
        
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        for i, logits in enumerate(scores):
            if i >= len(generated_ids):
                break
            
            token_id = generated_ids[i].item()
            
            if pad_token_id is not None and token_id == pad_token_id:
                break
            
            log_probs = F.log_softmax(logits[0], dim=-1)
            token_log_prob = log_probs[token_id].item()
            
            log_prob_sum += token_log_prob
            token_log_probs.append(token_log_prob)
            
            if eos_token_id is not None and token_id == eos_token_id:
                break
        
        return log_prob_sum, token_log_probs
    
    def get_last_log_prob(self) -> Optional[float]:
        return self.last_log_prob
    
    def get_last_token_log_probs(self) -> Optional[List[float]]:
        return self.last_token_log_probs
    
    def get_last_prompt(self) -> str:
        return self.last_prompt
    
    def get_last_input_ids(self) -> Optional[torch.Tensor]:
        return self.last_input_ids
    
    def get_last_generated_ids(self) -> Optional[torch.Tensor]:
        return self.last_generated_ids
    
    def get_last_raw_response(self) -> str:
        return self.last_raw_response


"""
PPO Trainer: Proximal Policy Optimization for cooperative/competitive communication.

PPO implementation kept consistent across other project experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import itertools

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

from .trajectory_buffer import TrajectoryBuffer, Trajectory
from .reward_function import RewardFunction
from ..game.protocol import GameMode


class CoopCommTrainer:
    """
    PPO Trainer for cooperative/competitive communication experiment.
    
    Uses Proximal Policy Optimization with:
    - Clipped surrogate objective
    - Value function baseline (critic)
    - Generalized Advantage Estimation (GAE)
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        learning_rate: float = 1e-5,
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        # PPO hyperparameters
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        target_kl: float = 0.02,
        # Other
        checkpoint_dir: str = "checkpoints/coop_comm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.use_4bit = use_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # PPO hyperparameters
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.target_kl = target_kl
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Components (initialized in load_model)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.value_head = None
        
        # Buffers
        self.trajectory_buffer = TrajectoryBuffer()
        self.reward_function = RewardFunction()
        
        # Metrics
        self.training_history: List[Dict[str, Any]] = []
        self.current_iteration = 0
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load model with LoRA and add value head."""
        print(f"Loading model: {self.model_name}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quantization_config = None
        compute_dtype = torch.float16
        
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype if self.use_4bit else torch.float32
        )
        
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Add value head (critic)
        self._add_value_head()
        
        # Optimizer (include both model and value head)
        all_params = itertools.chain(
            self.model.parameters(),
            self.value_head.parameters()
        )
        self.optimizer = AdamW(all_params, lr=self.learning_rate, weight_decay=0.01)
        
        return self.model, self.tokenizer
    
    def _add_value_head(self):
        """Add value head for PPO critic."""
        hidden_size = self.model.config.hidden_size
        
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.to(self.model.device)
        
        # Small initialization
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)
    
    def add_game_result(self, game_result: Any) -> None:
        """Add trajectories from a game result."""
        if hasattr(game_result, 'trajectories'):
            self.trajectory_buffer.add_trajectories(game_result.trajectories)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_gae * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            
            next_value = values[t]
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _recompute_log_probs_and_values(
        self,
        batch_trajectories: List[Trajectory],
        requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log probs and values with current model.
        
        Returns:
            Tuple of (new_log_probs, old_log_probs, values, entropy)
        """
        device = self.model.device
        
        new_log_probs_list = []
        old_log_probs_list = []
        values_list = []
        entropy_list = []
        
        for traj in batch_trajectories:
            if traj.input_ids is None or traj.generated_ids is None:
                # Fallback for invalid trajectory
                new_log_probs_list.append(torch.tensor(0.0, device=device))
                old_log_probs_list.append(torch.tensor(0.0, device=device))
                values_list.append(torch.tensor(0.0, device=device))
                entropy_list.append(torch.tensor(0.0, device=device))
                continue
            
            input_ids = traj.input_ids.to(device)
            generated_ids = traj.generated_ids.to(device)
            
            # Ensure both are 2D: (1, seq_len)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if generated_ids.dim() == 1:
                generated_ids = generated_ids.unsqueeze(0)
            
            # Full sequence = prompt + generated
            full_seq = torch.cat([input_ids, generated_ids], dim=1)
            
            # Attention mask
            attention_mask = torch.ones_like(full_seq)
            if self.tokenizer.pad_token_id is not None:
                attention_mask = (full_seq != self.tokenizer.pad_token_id).long()
            
            # Forward pass
            if requires_grad:
                outputs = self.model(
                    input_ids=full_seq,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            else:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=full_seq,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
            
            logits = outputs.logits
            prompt_len = input_ids.shape[1]
            
            # Compute value from last hidden state of prompt
            last_hidden = outputs.hidden_states[-1]
            last_prompt_hidden = last_hidden[:, prompt_len - 1, :].float()
            last_prompt_hidden = F.layer_norm(last_prompt_hidden, (last_prompt_hidden.size(-1),))
            
            value = self.value_head(last_prompt_hidden).squeeze()
            value = torch.clamp(value, -10.0, 10.0)
            
            if requires_grad:
                values_list.append(value)
            else:
                values_list.append(value.detach())
            
            # Compute log probs for generated tokens
            # generated_ids is 2D (1, seq_len), so use shape[1] for length
            num_generated = generated_ids.shape[1]
            token_log_probs = []
            token_entropies = []
            
            for i in range(num_generated):
                logit_pos = prompt_len - 1 + i
                if logit_pos >= logits.shape[1]:
                    break
                
                token_id = generated_ids[0, i].item()
                token_logits = logits[0, logit_pos, :].float()
                
                log_probs = F.log_softmax(token_logits, dim=-1)
                token_log_prob = log_probs[token_id]
                token_log_probs.append(token_log_prob)
                
                # Entropy
                probs = F.softmax(token_logits, dim=-1)
                entropy = -(probs * log_probs).sum()
                token_entropies.append(entropy)
            
            if token_log_probs:
                new_log_prob = torch.stack(token_log_probs).sum()
                entropy = torch.stack(token_entropies).mean()
            else:
                new_log_prob = torch.tensor(0.0, device=device)
                entropy = torch.tensor(0.0, device=device)
            
            # Old log prob from trajectory
            old_log_prob = torch.tensor(traj.log_prob, dtype=torch.float32, device=device)
            
            new_log_probs_list.append(new_log_prob)
            old_log_probs_list.append(old_log_prob)
            entropy_list.append(entropy)
        
        new_log_probs = torch.stack(new_log_probs_list)
        old_log_probs = torch.stack(old_log_probs_list)
        values = torch.stack(values_list)
        entropy = torch.stack(entropy_list)
        
        if not requires_grad:
            new_log_probs = new_log_probs.detach()
            values = values.detach()
            entropy = entropy.detach()
        
        return new_log_probs, old_log_probs, values, entropy
    
    def compute_ppo_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with clipping.
        
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Ratio
        log_ratio = new_log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )
        
        # Approximate KL for early stopping
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
        
        return total_loss, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl,
            "mean_ratio": ratio.mean().item()
        }
    
    def train_iteration(
        self,
        batch_size: int = 16,
        mini_batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Run one PPO training iteration.
        
        Args:
            batch_size: Logical batch size
            mini_batch_size: Physical batch size for gradient accumulation
            
        Returns:
            Training metrics
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if len(self.trajectory_buffer) == 0:
            return {"error": "No trajectories in buffer"}
        
        all_trajectories = self.trajectory_buffer.get_all()
        
        # count trajectories with valid tensors for debugging
        n_total = len(all_trajectories)
        n_with_tensors = sum(1 for t in all_trajectories if t.input_ids is not None and t.generated_ids is not None)
        print(f"DEBUG: Trajectories: {n_total} total, {n_with_tensors} with valid tensors")
        
        # Filter invalid trajectories
        valid_trajectories = [
            t for t in all_trajectories
            if np.isfinite(t.log_prob) and abs(t.log_prob) < 1000
        ]
        
        if not valid_trajectories:
            return {"error": "No valid trajectories"}
        
        # Sort by game and round for proper GAE
        valid_trajectories.sort(key=lambda t: (t.game_id, t.agent_id, t.round_number))
        
        rewards = [t.reward for t in valid_trajectories]
        
        # Compute initial values for GAE
        initial_values = []
        eval_batch_size = 8
        
        with torch.no_grad():
            for i in range(0, len(valid_trajectories), eval_batch_size):
                batch = valid_trajectories[i:i+eval_batch_size]
                _, _, values, _ = self._recompute_log_probs_and_values(batch)
                initial_values.extend(values.cpu().tolist())
        
        # Determine episode boundaries
        episode_keys = {}
        for i, traj in enumerate(valid_trajectories):
            key = (traj.game_id, traj.agent_id)
            if key not in episode_keys:
                episode_keys[key] = []
            episode_keys[key].append(i)
        
        dones = [False] * len(valid_trajectories)
        for indices in episode_keys.values():
            if indices:
                dones[indices[-1]] = True
        
        # Compute GAE
        advantages, returns = self.compute_gae(rewards, initial_values, dones)
        advantages = advantages.to(self.model.device)
        returns = returns.to(self.model.device)
        
        # PPO update loop
        all_metrics = []
        dataset_size = len(valid_trajectories)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                self.optimizer.zero_grad()
                
                # Process mini-batches
                for mini_start in range(0, len(batch_indices), mini_batch_size):
                    mini_end = min(mini_start + mini_batch_size, len(batch_indices))
                    mini_indices = batch_indices[mini_start:mini_end]
                    
                    mini_trajectories = [valid_trajectories[i] for i in mini_indices]
                    mini_advantages = advantages[mini_indices]
                    mini_returns = returns[mini_indices]
                    
                    # Forward pass with gradients
                    new_log_probs, old_log_probs, values, entropy = self._recompute_log_probs_and_values(
                        mini_trajectories, requires_grad=True
                    )
                    
                    # Compute loss
                    loss, loss_components = self.compute_ppo_loss(
                        new_log_probs, old_log_probs,
                        mini_advantages, values, mini_returns, entropy
                    )
                    
                    # Backward (accumulate gradients)
                    num_accumulation_steps = (len(batch_indices) + mini_batch_size - 1) // mini_batch_size
                    (loss / num_accumulation_steps).backward()
                    
                    all_metrics.append(loss_components)
                
                # Clip gradients and update
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Early stopping on KL
                if all_metrics and all_metrics[-1]["approx_kl"] > self.target_kl * 1.5:
                    break
        
        # Aggregate metrics
        if not all_metrics:
            return {"error": "No metrics computed"}
        
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        avg_metrics["num_trajectories"] = len(valid_trajectories)
        avg_metrics["avg_reward"] = np.mean(rewards)
        
        self.training_history.append(avg_metrics)
        self.current_iteration += 1
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Optional[Dict] = None) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save value head
        torch.save(self.value_head.state_dict(), checkpoint_path / "value_head.pt")
        
        # Save trainer state
        state = {
            "iteration": self.current_iteration,
            "training_history": self.training_history,
            "ppo_config": {
                "gamma": self.gamma,
                "lambda_gae": self.lambda_gae,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef
            },
            "metrics": metrics
        }
        
        with open(checkpoint_path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # Load model
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Load value head
        self._add_value_head()
        value_head_path = checkpoint_path / "value_head.pt"
        if value_head_path.exists():
            self.value_head.load_state_dict(torch.load(value_head_path))
        
        # Load state
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
            self.current_iteration = state.get("iteration", 0)
            self.training_history = state.get("training_history", [])
        
        # Reinit optimizer
        all_params = itertools.chain(self.model.parameters(), self.value_head.parameters())
        self.optimizer = AdamW(all_params, lr=self.learning_rate)
    
    def clear_buffer(self) -> None:
        """Clear trajectory buffer."""
        self.trajectory_buffer.clear()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history:
            return {}
        
        recent = self.training_history[-10:]
        
        return {
            "total_iterations": len(self.training_history),
            "recent_avg_policy_loss": np.mean([h.get("policy_loss", 0) for h in recent]),
            "recent_avg_value_loss": np.mean([h.get("value_loss", 0) for h in recent]),
            "recent_avg_reward": np.mean([h.get("avg_reward", 0) for h in recent]),
            "buffer_size": len(self.trajectory_buffer)
        }

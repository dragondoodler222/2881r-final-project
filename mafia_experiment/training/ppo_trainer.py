"""
PPO Trainer: Proximal Policy Optimization for Mafia agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import os

from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper
)

from .model_manager import ModelManager
from .reward_function import RewardFunction
from .trajectory_buffer import TrajectoryBuffer, Trajectory


from torch.distributions import Categorical

class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) trainer for self-play learning

    PPO is more stable than REINFORCE and is the standard for LLM fine-tuning (RLHF).
    Uses clipped objective and value function for variance reduction.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        reward_function: RewardFunction,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,  # GAE parameter
        clip_epsilon: float = 0.2,  # PPO clipping parameter
        value_coef: float = 0.5,  # Value loss coefficient
        entropy_coef: float = 0.01,  # Entropy bonus coefficient
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,  # Number of PPO update epochs per batch
        target_kl: float = 0.02,  # Target KL divergence for early stopping
        checkpoint_dir: str = "checkpoints"
    ):
        self.model_manager = model_manager
        self.reward_function = reward_function
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

        # Training buffer
        self.trajectory_buffer = TrajectoryBuffer()

        # Use already-loaded model from model_manager
        if model_manager.model is None or model_manager.tokenizer is None:
            raise ValueError("ModelManager must have model loaded before creating PPOTrainer. Call model_manager.load_model_with_lora() first.")

        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer

        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        # Add value head to model (for critic)
        self._add_value_head()

        # Optimizer (include both model and value head parameters)
        # Combine model parameters with value head parameters
        import itertools
        all_parameters = itertools.chain(
            self.model.parameters(),
            self.value_head.parameters()
        )
        self.optimizer = AdamW(
            all_parameters,
            lr=learning_rate,
            weight_decay=0.01
        )

        # Training metrics
        self.training_history = []
        self.current_iteration = 0

        # Debug options
        self.debug_kl = bool(int(os.environ.get("PPO_DEBUG_KL", "0")))
        self._debug_log_diffs: List[float] = []
        gen_config = getattr(self.model, "generation_config", None)
        if gen_config is not None:
            self._base_generation_top_k = getattr(gen_config, "top_k", 0) or 0
            self._base_generation_top_p = getattr(gen_config, "top_p", 1.0)
        else:
            self._base_generation_top_k = 0
            self._base_generation_top_p = 1.0
        self._warper_cache: Dict[Tuple[float, int, Optional[float]], Optional[LogitsProcessorList]] = {}

    def _add_value_head(self):
        """Add a value head to the model for the critic"""
        # Get hidden size from model config (works for Llama 2, Mistral, etc.)
        hidden_size = self.model.config.hidden_size

        # Create value head: linear layer that maps hidden states to scalar value
        self.value_head = nn.Linear(hidden_size, 1)

        # Move to same device as model
        self.value_head.to(self.model.device)

        # Initialize weights (small random initialization)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.constant_(self.value_head.bias, 0.0)

    def add_game_trajectories(
        self,
        trajectories: List[Trajectory],
        game_result: Dict[str, Any]
    ) -> None:
        """
        Add trajectories from a completed game

        Args:
            trajectories: List of trajectory steps from the game
            game_result: Game outcome information
        """
        # Assign rewards to trajectories
        trajectories_with_rewards = self.reward_function.assign_rewards_to_trajectories(
            [t.__dict__ if hasattr(t, '__dict__') else t for t in trajectories],
            game_result["game_state"],
            game_result["winner"],
            game_result["total_rounds"]
        )

        # Add to buffer
        for traj_dict in trajectories_with_rewards:
            if isinstance(traj_dict, dict):
                traj = Trajectory(**traj_dict)
            else:
                traj = traj_dict
            self.trajectory_buffer.add_trajectory(traj)

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of episode termination flags

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
            # # Clip advantages to prevent extreme updates
            # advantages = advantages.clamp(-6.0, 6.0)

        return advantages, returns

    def _get_logits_warper_for_temperature(self, temperature: float) -> Optional[LogitsProcessorList]:
        """Cache and return HuggingFace logits warpers for a specific temperature."""
        temp = float(temperature) if temperature and temperature > 0 else 1.0
        top_k = self._base_generation_top_k
        top_p = self._base_generation_top_p
        top_p_key = round(top_p, 3) if isinstance(top_p, float) else top_p
        key = (round(temp, 3), int(top_k), top_p_key)

        if key not in self._warper_cache:
            processors = LogitsProcessorList()
            if temp != 1.0:
                processors.append(TemperatureLogitsWarper(temp))
            if top_k and top_k > 0:
                processors.append(TopKLogitsWarper(top_k))
            if top_p is not None and isinstance(top_p, float) and 0.0 < top_p < 1.0:
                processors.append(TopPLogitsWarper(top_p))

            self._warper_cache[key] = processors

        return self._warper_cache[key]

    def _apply_sampling_warpers(
        self,
        logits: torch.Tensor,
        temperature: float,
        input_ids: torch.Tensor,
        cached_warper: Optional[LogitsProcessorList] = None
    ) -> torch.Tensor:
        """Apply the exact sampling warpers used during generation."""
        warper = cached_warper if cached_warper is not None else self._get_logits_warper_for_temperature(temperature)
        adjusted = logits.float()

        if warper is None or len(warper) == 0:
            return adjusted

        squeeze = False
        if adjusted.dim() == 1:
            adjusted = adjusted.unsqueeze(0)
            squeeze = True

        current_input_ids = input_ids
        if current_input_ids.dim() == 1:
            current_input_ids = current_input_ids.unsqueeze(0)

        warped = warper(current_input_ids, adjusted)

        if squeeze:
            warped = warped.squeeze(0)

        return warped

    def _recompute_log_probs_and_values(
        self,
        batch_trajectories: List[Trajectory],
        requires_grad: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log probabilities and values using current model weights

        This is the CORE of PPO - we need to recompute these values with the
        updated model to compute the policy ratio for the PPO objective.

        Args:
            batch_trajectories: List of trajectories with stored prompts
            requires_grad: Whether to compute gradients (True for training, False for evaluation)

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        device = self.model.device
        all_log_probs = []
        all_values = []
        all_entropies = []

        # Ensure tokenizer uses left padding for correct position IDs with causal models
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        for trajectory in batch_trajectories:
            # Get stored prompt and generated tokens
            prompt = trajectory.prompt
            generated_ids = trajectory.generated_ids  # The tokens that were actually generated
            temperature = getattr(trajectory, 'temperature', 1.0)
            warper = self._get_logits_warper_for_temperature(temperature)

            if not prompt:
                # Fallback: use stored log_prob if no prompt available
                # Convert to tensor on device to ensure consistency
                safe_log_prob = trajectory.log_prob if trajectory.log_prob != 0.0 else -100.0
                all_log_probs.append(torch.tensor(safe_log_prob, dtype=torch.float32, device=device))
                all_values.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                all_entropies.append(torch.tensor(0.1, dtype=torch.float32, device=device))
                continue

            # For PPO, we need to run the model on the full sequence (prompt + generated tokens)
            # to get the logits at the positions where tokens were generated
            if generated_ids is not None and len(generated_ids) > 0:
                # Use stored input_ids if available, otherwise tokenize
                if trajectory.input_ids is not None:
                    input_ids = trajectory.input_ids.to(device)
                else:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024
                    )
                    input_ids = inputs.input_ids.to(device)

                # Ensure input_ids is 2D (batch_size, seq_len)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                # Concatenate prompt tokens with generated tokens
                generated_ids_device = generated_ids.to(device).unsqueeze(0)  # Add batch dim
                
                full_sequence = torch.cat([input_ids, generated_ids_device], dim=1)

                # Create attention mask (0 for padding, 1 for real tokens)
                if self.tokenizer.pad_token_id is not None:
                    attention_mask = (full_sequence != self.tokenizer.pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(full_sequence)

                # CRITICAL FIX: Ensure attention mask is contiguous and on correct device
                attention_mask = attention_mask.to(device).contiguous()

                # Forward pass with full sequence
                if requires_grad:
                    outputs = self.model(
                        input_ids=full_sequence,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                else:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=full_sequence,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            use_cache=False  # Save memory by not caching KV
                        )
                
                # Get prompt length
                prompt_len = input_ids.shape[1]
            else:
                # Fallback: just use prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                input_ids = inputs.input_ids
                prompt_len = input_ids.shape[1]

                # Forward pass through model (with or without gradients)
                if requires_grad:
                    outputs = self.model(**inputs, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

            # Get last hidden state for value computation
            # Shape: (batch_size=1, seq_len, hidden_size)
            last_hidden_state = outputs.hidden_states[-1]

            # For value: use the last token of the PROMPT (state value V(s))
            # Shape: (1, hidden_size)
            # Note: prompt_len - 1 is the index of the last token of the prompt
            last_prompt_hidden = last_hidden_state[:, prompt_len - 1, :].float()  # Cast to float32 for stability

            # FIXED: Normalize hidden states to prevent value head instability
            # Quantized models can have large magnitude hidden states
            last_prompt_hidden = F.layer_norm(
                last_prompt_hidden,
                (last_prompt_hidden.size(-1),)
            )

            # Compute value using value head
            # Shape: (1, 1) -> scalar
            value = self.value_head(last_prompt_hidden).squeeze()

            # FIXED: Clamp value to prevent extreme values
            value = torch.clamp(value, min=-10.0, max=10.0)

            if requires_grad:
                # Keep as tensor for gradient computation
                all_values.append(value)
            else:
                all_values.append(value.detach())

            # Recompute log probability of the action that was taken
            # We need to compute the log prob of the EXACT tokens that were generated
            if generated_ids is not None and len(generated_ids) > 0 and trajectory.input_ids is not None:
                # Get logits for all positions
                # Shape: (1, seq_len, vocab_size)
                logits = outputs.logits

                # Move generated_ids to device
                generated_ids_device = generated_ids.to(device)

                # The logits at position i predict the token at position i+1
                # So to get the log prob of generated tokens, we look at logits from
                # positions (len(input_ids)-1) to (len(full_sequence)-2)
                num_generated = len(generated_ids_device)

                # FIXED: Accumulate in lists, not tensors (proper PyTorch pattern)
                token_log_probs_list = []
                token_entropies_list = []

                # Create a mask for non-padding tokens in generated_ids
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
                eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -100

                for i in range(num_generated):
                    # Logit position that predicts the i-th generated token
                    logit_position = prompt_len - 1 + i
                    current_input_ids = full_sequence[:, :prompt_len + i]

                    if logit_position < logits.shape[1]:
                        # Get token id
                        token_id = generated_ids_device[i].item()

                        # Handle EOS and Padding
                        if token_id == pad_token_id:
                            continue

                        token_logits = logits[0, logit_position, :].float()  # Cast to float32 for stability
                        token_logits = self._apply_sampling_warpers(
                            token_logits,
                            temperature,
                            current_input_ids,
                            cached_warper=warper
                        )
                        token_log_probs = F.log_softmax(token_logits, dim=-1)

                        # Get log prob of the token that was actually generated
                        token_log_probs_list.append(token_log_probs[token_id])

                        # Compute entropy safely - FIXED: Avoid NaN from 0 * -inf
                        token_probs = F.softmax(token_logits, dim=-1)
                        # Clamp to avoid log(0) = -inf
                        token_probs_safe = torch.clamp(token_probs, min=1e-10)
                        token_log_probs_safe = torch.log(token_probs_safe)
                        token_entropy = -(token_probs * token_log_probs_safe).sum(dim=-1)
                        token_entropies_list.append(token_entropy)

                        if token_id == eos_token_id:
                            break

                # Sum log prob across all tokens (standard for PPO)
                if len(token_log_probs_list) > 0:
                    # FIXED: Proper tensor stacking and summing
                    new_log_prob = torch.stack(token_log_probs_list).sum()
                    sum_entropy = torch.stack(token_entropies_list).sum()
                else:
                    # Empty sequence - use stored log_prob
                    safe_log_prob = trajectory.log_prob if trajectory.log_prob != 0.0 else -100.0
                    new_log_prob = torch.tensor(safe_log_prob, dtype=torch.float32, device=device)
                    sum_entropy = torch.tensor(0.1, dtype=torch.float32, device=device)  # Small positive entropy

                all_log_probs.append(new_log_prob)
                all_entropies.append(sum_entropy)
                if self.debug_kl and not requires_grad:
                    self._debug_log_diffs.append(new_log_prob.item() - float(trajectory.log_prob))
            else:
                # Fallback: use stored log_prob if no generated_ids available
                safe_log_prob = trajectory.log_prob if trajectory.log_prob != 0.0 else -100.0
                all_log_probs.append(torch.tensor(safe_log_prob, dtype=torch.float32, device=device))

                # Compute entropy from last position - FIXED: Safe computation
                last_logits = outputs.logits[:, -1, :].float()  # Cast to float32
                last_logits = self._apply_sampling_warpers(
                    last_logits,
                    temperature,
                    input_ids,
                    cached_warper=warper
                )
                last_probs = F.softmax(last_logits, dim=-1)
                # FIXED: Clamp to avoid log(0) = -inf
                last_probs_safe = torch.clamp(last_probs, min=1e-10)
                last_log_probs_safe = torch.log(last_probs_safe)
                entropy = -(last_probs * last_log_probs_safe).sum(dim=-1)
                all_entropies.append(entropy)
                if self.debug_kl and not requires_grad:
                    self._debug_log_diffs.append(float(safe_log_prob) - float(trajectory.log_prob))

            # Free intermediate tensors to reduce memory pressure
            del outputs
            if 'full_sequence' in dir():
                del full_sequence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Stack tensors
        log_probs_tensor = torch.stack(all_log_probs)
        values_tensor = torch.stack(all_values)
        entropy_tensor = torch.stack(all_entropies)

        # FIXED: Added root cause fixes above, so NaNs should not occur
        # Final safety check (should rarely trigger now)
        if torch.isnan(log_probs_tensor).any() or torch.isnan(values_tensor).any() or torch.isnan(entropy_tensor).any():
            # Replace NaNs with zeros or small values to prevent crash, but this indicates a problem
            log_probs_tensor = torch.nan_to_num(log_probs_tensor, nan=-100.0, neginf=-100.0)
            values_tensor = torch.nan_to_num(values_tensor, nan=0.0)
            entropy_tensor = torch.nan_to_num(entropy_tensor, nan=0.0)
            
            print("WARNING: NaNs detected in _recompute_log_probs_and_values outputs! Replaced with safe values.")

        if not requires_grad:
            log_probs_tensor = log_probs_tensor.detach()
            values_tensor = values_tensor.detach()
            entropy_tensor = entropy_tensor.detach()

        return log_probs_tensor, values_tensor, entropy_tensor

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with clipping

        Args:
            log_probs: New log probabilities
            old_log_probs: Old log probabilities (from data collection)
            advantages: Advantage estimates
            values: Value predictions
            returns: Target returns
            entropy: Policy entropy
            confidence: Parsing confidence weights (0.0 to 1.0)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Ratio between new and old policy
        
        # Filter out non-finite values (NaNs or Infs)
        finite_mask = torch.isfinite(log_probs) & torch.isfinite(old_log_probs)
        
        if not finite_mask.all():
            # Only keep valid entries
            log_probs = log_probs[finite_mask]
            old_log_probs = old_log_probs[finite_mask]
            advantages = advantages[finite_mask]
            values = values[finite_mask]
            returns = returns[finite_mask]
            entropy = entropy[finite_mask]
            if confidence is not None:
                confidence = confidence[finite_mask]
                
        if len(log_probs) == 0:
             return torch.tensor(0.0, device=self.model.device, requires_grad=True), {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0
            }
        
        # Safe log-ratio computation
        log_ratio = log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        
        # Standard PPO loss (unweighted)
        policy_loss_unweighted = -torch.min(surr1, surr2)
        
        # Apply confidence weighting if provided
        if confidence is not None:
            # Weight the policy loss by confidence
            # Low confidence -> low weight -> less gradient update
            policy_loss = (policy_loss_unweighted * confidence).mean()
        else:
            policy_loss = policy_loss_unweighted.mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropy.mean()

        # Calculate approximate KL divergence for logging/early stopping
        with torch.no_grad():
            # http://joschu.net/blog/kl-approx.html
            # Use a more stable KL approximation: 0.5 * (log_ratio)^2
            # This is less sensitive to extreme ratios than (ratio - 1) - log_ratio
            approx_kl = 0.5 * (log_ratio ** 2).mean()
            # per_sample_kl = 0.5 * (log_ratio ** 2)
            # per_sample_kl = per_sample_kl.clamp(max=0.5)  # Clamp to disallow a few extreme sample KL values from dominating KL calc
            # approx_kl = per_sample_kl.mean()

        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        loss_components = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl.item()
        }

        return total_loss, loss_components

    def train_iteration(
        self,
        batch_size: int = 32,  # Logical batch size
        mini_batch_size: int = 4  # Physical batch size
    ) -> Dict[str, float]:
        """
        Run one PPO training iteration on collected trajectories

        Args:
            batch_size: Logical batch size for updates
            mini_batch_size: Physical batch size to fit in memory

        Returns:
            Dictionary of training metrics
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(self.trajectory_buffer) == 0:
            return {"error": "No trajectories in buffer"}

        # Get ALL trajectories from buffer (PPO is on-policy)
        raw_trajectories = self.trajectory_buffer.get_all()
        
        if not raw_trajectories:
            return {"error": "Empty buffer"}

        # --- FILTERING STEP: Remove trajectories with invalid log_probs ---
        all_trajectories = []
        for t in raw_trajectories:
            # Check for inf/nan in stored log_prob
            if np.isfinite(t.log_prob) and abs(t.log_prob) < 1000:
                all_trajectories.append(t)
        
        dropped_count = len(raw_trajectories) - len(all_trajectories)
        if dropped_count > 0:
            print(f"Dropped {dropped_count} trajectories with invalid stored log_probs")

        if not all_trajectories:
            return {"error": "No valid trajectories after filtering"}

        # Sort trajectories by game, agent, and round to ensure temporal order for GAE
        # This is CRITICAL for correct advantage estimation
        all_trajectories.sort(key=lambda t: (t.game_id, t.agent_id, t.round_number))

        # Extract data from trajectories
        old_log_probs = torch.tensor([t.log_prob for t in all_trajectories], dtype=torch.float32)
        
        # Sanitize old_log_probs - REMOVED to allow raw values
        # old_log_probs = torch.nan_to_num(...)
        
        # --- CRITICAL FIX: Clamp old_log_probs to avoid extreme values ---
        # REMOVED: We now filter invalid values in compute_ppo_loss instead of clamping
        # old_log_probs = torch.clamp(old_log_probs, min=-20.0, max=20.0)
        
        rewards = [t.reward for t in all_trajectories]
        
        # Extract confidence scores (default to 1.0 if not present)
        confidences = torch.tensor(
            [getattr(t, 'parsing_confidence', 1.0) for t in all_trajectories], 
            dtype=torch.float32
        )

        # ============================================================================
        # REAL PPO IMPLEMENTATION: Compute values from current model for GAE
        # ============================================================================

        # First pass: compute initial values for GAE computation
        # We process in chunks to avoid OOM if buffer is large
        initial_values_list = []
        valid_indices = [] # Keep track of indices that pass recomputation
        
        eval_batch_size = 16  # Small batch size for evaluation
        if self.debug_kl:
            self._debug_log_diffs = []
        
        with torch.no_grad():
            for i in range(0, len(all_trajectories), eval_batch_size):
                batch = all_trajectories[i:i+eval_batch_size]
                batch_log_probs, values, _ = self._recompute_log_probs_and_values(batch)
                
                # Check for validity of recomputed values
                for j, (val, lp) in enumerate(zip(values, batch_log_probs)):
                    if torch.isfinite(val) and torch.isfinite(lp):
                        initial_values_list.append(val.item())
                        valid_indices.append(i + j)
        
        # --- SECOND FILTERING STEP: Remove trajectories where recomputation failed ---
        if len(valid_indices) < len(all_trajectories):
            print(f"Dropped {len(all_trajectories) - len(valid_indices)} trajectories due to recomputation failure")
            all_trajectories = [all_trajectories[i] for i in valid_indices]
            old_log_probs = old_log_probs[valid_indices]
            rewards = [rewards[i] for i in valid_indices]
            confidences = confidences[valid_indices]
            # initial_values_list is already filtered

        if self.debug_kl and self._debug_log_diffs:
            diffs = np.array(self._debug_log_diffs)
            # Filter out infs/nans for stats
            finite_diffs = diffs[np.isfinite(diffs)]
            if len(finite_diffs) > 0:
                print(
                    f"[PPO DEBUG] old/new log_prob diff -> mean {finite_diffs.mean():.2f}, std {finite_diffs.std():.2f}, "
                    f"min {finite_diffs.min():.2f}, max {finite_diffs.max():.2f}, "
                    f"infs/nans: {len(diffs) - len(finite_diffs)}"
                )
            else:
                print(f"[PPO DEBUG] old/new log_prob diff -> all infs/nans (count: {len(diffs)})")

        # Determine episode boundaries (dones)
        # Group by game_id and agent_id, mark last trajectory in each episode as done
        episode_keys = {}
        for i, traj in enumerate(all_trajectories):
            key = (traj.game_id, traj.agent_id)
            if key not in episode_keys:
                episode_keys[key] = []
            episode_keys[key].append(i)

        dones = [False] * len(all_trajectories)
        for indices in episode_keys.values():
            if indices:
                dones[indices[-1]] = True

        # Compute GAE using initial values
        advantages, returns = self.compute_gae(rewards, initial_values_list, dones)

        # Move to device
        device = self.model.device
        old_log_probs = old_log_probs.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        confidences = confidences.to(device)

        # PPO update loop
        all_metrics = []
        
        # Create indices for shuffling
        dataset_size = len(all_trajectories)
        indices = np.arange(dataset_size)
        
        # Track KL for early stopping
        stop_early = False

        for epoch in range(self.ppo_epochs):
            if stop_early:
                break

            # Shuffle for each epoch
            np.random.shuffle(indices)
            
            # Iterate over logical batches
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                current_batch_size = len(batch_indices)
                
                # Zero gradients at start of logical batch
                self.optimizer.zero_grad()
                
                # Accumulate gradients over mini-batches
                batch_kls = []
                
                for mini_start in range(0, current_batch_size, mini_batch_size):
                    mini_end = min(mini_start + mini_batch_size, current_batch_size)
                    mini_indices = batch_indices[mini_start:mini_end]
                    
                    # Get mini-batch data
                    mini_trajectories = [all_trajectories[i] for i in mini_indices]
                    mini_old_log_probs = old_log_probs[mini_indices]
                    mini_advantages = advantages[mini_indices]
                    mini_returns = returns[mini_indices]
                    mini_confidences = confidences[mini_indices]
                    
                    # REAL IMPLEMENTATION: Forward pass through model with gradients
                    # Recompute log probs, values, and entropy with current model weights
                    new_log_probs, new_values, entropy = self._recompute_log_probs_and_values(
                        mini_trajectories,
                        requires_grad=True
                    )

                    # Compute PPO loss
                    loss, loss_components = self.compute_ppo_loss(
                        new_log_probs,
                        mini_old_log_probs,
                        mini_advantages,
                        new_values,
                        mini_returns,
                        entropy,
                        confidence=mini_confidences
                    )
                    
                    # Track KL
                    batch_kls.append(loss_components["approx_kl"])
                    
                    # Scale loss for gradient accumulation
                    loss_scale = len(mini_indices) / current_batch_size
                    scaled_loss = loss * loss_scale
                    
                    # Backward pass (accumulates gradients)
                    scaled_loss.backward()
                    
                    # Store metrics (unscaled)
                    all_metrics.append(loss_components)

                # Check for early stopping based on KL
                mean_kl = np.mean(batch_kls)
                
                # Always print KL if debug flag is on
                if os.environ.get("PPO_DEBUG_KL", "0") == "1":
                    print(f"[PPO DEBUG] Epoch {epoch} Batch KL: {mean_kl:.4f}")

                if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch} due to KL divergence {mean_kl:.4f} > {1.5 * self.target_kl:.4f}")
                    stop_early = True
                    # Don't step optimizer if KL is too high
                    self.optimizer.zero_grad()
                    break

                # Gradient clipping (clip both model and value head)
                import itertools
                all_params = itertools.chain(
                    self.model.parameters(),
                    self.value_head.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    all_params,
                    self.max_grad_norm
                )

                # Update weights
                self.optimizer.step()

                # Clear GPU cache to free fragmented memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Average metrics across PPO epochs
        if not all_metrics:
            return {"error": "No metrics collected (early stopping before first update?)"}

        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }

        # Add additional metrics
        avg_metrics.update({
            "iteration": self.current_iteration,
            "mean_reward": np.mean(rewards),
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item(),
            "num_trajectories": len(all_trajectories),
            "ppo_epochs": self.ppo_epochs
        })

        self.training_history.append(avg_metrics)
        self.current_iteration += 1
        
        # Clear buffer after training (on-policy)
        self.clear_buffer()

        return avg_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save training checkpoint

        Args:
            epoch: Current training epoch
            metrics: Optional metrics to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{epoch}"

        # Save model
        self.model_manager.save_checkpoint(
            save_path=str(self.checkpoint_dir),
            epoch=epoch,
            metrics=metrics
        )

        # Save value head
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        value_head_path = checkpoint_path / "value_head.pt"
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save trainer state
        trainer_state = {
            "iteration": self.current_iteration,
            "training_history": self.training_history,
            "ppo_config": {
                "gamma": self.gamma,
                "lambda_gae": self.lambda_gae,
                "clip_epsilon": self.clip_epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef
            }
        }

        state_path = checkpoint_path / "trainer_state.json"
        with open(state_path, "w") as f:
            json.dump(trainer_state, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, load_model: bool = True) -> None:
        """Load training checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if load_model:
            # Load model
            self.model, self.tokenizer = self.model_manager.load_checkpoint(
                str(checkpoint_path)
            )

        # Load value head if available
        value_head_path = checkpoint_path / "value_head.pt"
        if value_head_path.exists():
            self.value_head.load_state_dict(torch.load(value_head_path))
            print("Loaded value head from checkpoint")
        else:
            print("WARNING: No value head found in checkpoint, using initialized weights")

        # Reinitialize optimizer (include both model and value head)
        import itertools
        all_parameters = itertools.chain(
            self.model.parameters(),
            self.value_head.parameters()
        )
        self.optimizer = AdamW(
            all_parameters,
            lr=self.optimizer.param_groups[0]["lr"]
        )

        # Load trainer state if available
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                trainer_state = json.load(f)

            self.current_iteration = trainer_state.get("iteration", 0)
            self.training_history = trainer_state.get("training_history", [])

            print(f"Loaded trainer state from iteration {self.current_iteration}")

    def clear_buffer(self) -> None:
        """Clear the trajectory buffer"""
        self.trajectory_buffer.clear()

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_history:
            return {}

        recent_history = self.training_history[-10:]  # Last 10 iterations

        return {
            "total_iterations": len(self.training_history),
            "current_iteration": self.current_iteration,
            "recent_mean_policy_loss": np.mean([h["policy_loss"] for h in recent_history]),
            "recent_mean_value_loss": np.mean([h["value_loss"] for h in recent_history]),
            "recent_mean_reward": np.mean([h["mean_reward"] for h in recent_history]),
            "buffer_size": len(self.trajectory_buffer)
        }

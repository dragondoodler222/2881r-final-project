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
import re
import itertools

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
        # Force-disable top-k / top-p truncation so log probs remain finite.
        # Sampling still uses temperature for stochasticity, but every token
        # retains non-zero mass which keeps PPO ratios well-defined.
        self._base_generation_top_k = 0
        self._base_generation_top_p = 1.0
        if gen_config is not None:
            prior_top_k = getattr(gen_config, "top_k", 0) or 0
            prior_top_p = getattr(gen_config, "top_p", 1.0)
            if prior_top_k not in (None, 0):
                print(f"PPOTrainer: overriding generation top_k {prior_top_k} -> 0 for stability")
            if prior_top_p is not None and prior_top_p < 1.0:
                print(f"PPOTrainer: overriding generation top_p {prior_top_p} -> 1.0 for stability")
            gen_config.top_k = 0
            gen_config.top_p = 1.0
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recompute log probabilities and values using current model weights

        Returns:
            Tuple of (log_probs, old_log_probs, values, entropy, masks)
            log_probs: (batch_size, max_seq_len)
            old_log_probs: (batch_size, max_seq_len)
            values: (batch_size, 1)
            entropy: (batch_size, max_seq_len)
            masks: (batch_size, max_seq_len)
        """
        device = self.model.device
        all_log_probs_list = []
        all_old_log_probs_list = []
        all_values = []
        all_entropies_list = []
        all_masks_list = []

        # Ensure tokenizer uses left padding for correct position IDs with causal models
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'

        for trajectory in batch_trajectories:
            # Get stored prompt and generated tokens
            prompt = trajectory.prompt
            generated_ids = trajectory.generated_ids  # The tokens that were actually generated
            temperature = getattr(trajectory, 'temperature', 1.0)
            warper = self._get_logits_warper_for_temperature(temperature)
            
            # Get old token log probs
            old_token_log_probs = getattr(trajectory, 'token_log_probs', None)

            if not prompt:
                # Fallback: skip or use dummy
                # We need to append something to keep batch alignment
                all_log_probs_list.append(torch.tensor([0.0], device=device))
                all_old_log_probs_list.append(torch.tensor([0.0], device=device))
                all_values.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                all_entropies_list.append(torch.tensor([0.0], device=device))
                all_masks_list.append(torch.tensor([0.0], device=device))
                continue

            # Create region mask
            # We use the stored text (cot) and generated_ids
            # If generated_ids is None, we can't mask properly
            if generated_ids is not None and len(generated_ids) > 0:
                region_mask = self._create_region_mask(trajectory.cot, generated_ids, trajectory.phase)
            else:
                region_mask = torch.tensor([0.0], device=device)

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
                            # If we hit padding, we stop collecting
                            # But we need to match lengths.
                            # We'll just append 0.0 and mask it out later if needed
                            # But usually generated_ids are stripped of padding in Trajectory?
                            # Trajectory stores stripped_gen_ids.
                            # So we shouldn't hit padding unless it's EOS.
                            pass

                        token_logits = logits[0, logit_position, :].float()  # Cast to float32 for stability
                        warped_logits = self._apply_sampling_warpers(
                            token_logits,
                            temperature,
                            current_input_ids,
                            cached_warper=warper
                        )
                        token_log_probs = F.log_softmax(warped_logits, dim=-1)
                        token_log_prob = token_log_probs[token_id]

                        # If truncation ever produced -inf (should not happen now),
                        # fall back to temperature-only distribution to keep PPO finite.
                        if not torch.isfinite(token_log_prob):
                            safe_logits = token_logits
                            if temperature and temperature != 1.0:
                                safe_logits = TemperatureLogitsWarper(float(temperature))(current_input_ids, safe_logits)
                            token_log_probs = F.log_softmax(safe_logits, dim=-1)
                            token_log_prob = token_log_probs[token_id]

                        # Get log prob of the token that was actually generated
                        token_log_probs_list.append(token_log_prob)

                        # Compute entropy safely - FIXED: Avoid NaN from 0 * -inf
                        token_probs = F.softmax(token_logits, dim=-1)
                        # Clamp to avoid log(0) = -inf
                        token_probs_safe = torch.clamp(token_probs, min=1e-10)
                        token_log_probs_safe = torch.log(token_probs_safe)
                        token_entropy = -(token_probs * token_log_probs_safe).sum(dim=-1)
                        token_entropies_list.append(token_entropy)

                        if token_id == eos_token_id:
                            break

                # Stack lists to tensors
                if len(token_log_probs_list) > 0:
                    new_log_probs = torch.stack(token_log_probs_list)
                    new_entropies = torch.stack(token_entropies_list)
                    if getattr(self, "debug_kl", False) and not requires_grad:
                        try:
                            stored_log_prob = float(getattr(trajectory, "log_prob", 0.0))
                            recomputed_sum = float(new_log_probs.sum().detach().cpu())
                            self._debug_log_diffs.append(recomputed_sum - stored_log_prob)
                        except Exception:
                            pass
                else:
                    new_log_probs = torch.tensor([0.0], device=device)
                    new_entropies = torch.tensor([0.0], device=device)

                # Handle old log probs
                if old_token_log_probs is not None:
                    # Use stored per-token log probs
                    # Truncate or pad to match new_log_probs length if necessary (should match)
                    old_log_probs = torch.tensor(old_token_log_probs, dtype=torch.float32, device=device)
                    if len(old_log_probs) > len(new_log_probs):
                        old_log_probs = old_log_probs[:len(new_log_probs)]
                    elif len(old_log_probs) < len(new_log_probs):
                        # This shouldn't happen if generated_ids matches
                        padding = torch.zeros(len(new_log_probs) - len(old_log_probs), device=device)
                        old_log_probs = torch.cat([old_log_probs, padding])
                else:
                    # Fallback: distribute scalar log prob uniformly? No, that's bad.
                    # Fallback: use new log probs (KL=0)
                    old_log_probs = new_log_probs.detach()

                # Ensure mask matches length
                if len(region_mask) > len(new_log_probs):
                    region_mask = region_mask[:len(new_log_probs)]
                elif len(region_mask) < len(new_log_probs):
                    padding = torch.zeros(len(new_log_probs) - len(region_mask), device=device)
                    region_mask = torch.cat([region_mask, padding])

                all_log_probs_list.append(new_log_probs)
                all_old_log_probs_list.append(old_log_probs)
                all_entropies_list.append(new_entropies)
                all_masks_list.append(region_mask)

            else:
                # Fallback
                all_log_probs_list.append(torch.tensor([0.0], device=device))
                all_old_log_probs_list.append(torch.tensor([0.0], device=device))
                all_entropies_list.append(torch.tensor([0.0], device=device))
                all_masks_list.append(torch.tensor([0.0], device=device))

            # Free intermediate tensors to reduce memory pressure
            del outputs
            if 'full_sequence' in dir():
                del full_sequence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Pad sequences to max length in batch
        from torch.nn.utils.rnn import pad_sequence
        
        log_probs_tensor = pad_sequence(all_log_probs_list, batch_first=True, padding_value=0.0).to(device)
        old_log_probs_tensor = pad_sequence(all_old_log_probs_list, batch_first=True, padding_value=0.0).to(device)
        entropy_tensor = pad_sequence(all_entropies_list, batch_first=True, padding_value=0.0).to(device)
        masks_tensor = pad_sequence(all_masks_list, batch_first=True, padding_value=0.0).to(device)
        values_tensor = torch.stack(all_values).to(device)

        if not requires_grad:
            log_probs_tensor = log_probs_tensor.detach()
            old_log_probs_tensor = old_log_probs_tensor.detach()
            values_tensor = values_tensor.detach()
            entropy_tensor = entropy_tensor.detach()
            masks_tensor = masks_tensor.detach()

        return log_probs_tensor, old_log_probs_tensor, values_tensor, entropy_tensor, masks_tensor

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO loss with clipping and region masking

        Args:
            log_probs: New log probabilities (B, L)
            old_log_probs: Old log probabilities (B, L)
            advantages: Advantage estimates (B, 1)
            values: Value predictions (B, 1)
            returns: Target returns (B, 1)
            entropy: Policy entropy (B, L)
            masks: Region masks (B, L)
            confidence: Parsing confidence weights (0.0 to 1.0)

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Ratio between new and old policy
        
        # Filter out non-finite values (NaNs or Infs)
        finite_mask = torch.isfinite(log_probs) & torch.isfinite(old_log_probs)
        
        if not finite_mask.all():
            # Replace non-finite values with safe values (0 log prob = 1.0 prob, but log prob should be negative)
            # We use -100.0 for log prob (very small prob)
            log_probs = torch.nan_to_num(log_probs, nan=-100.0, neginf=-100.0, posinf=10.0)
            old_log_probs = torch.nan_to_num(old_log_probs, nan=-100.0, neginf=-100.0, posinf=10.0)
                
        if len(log_probs) == 0:
             return torch.tensor(0.0, requires_grad=True, device=log_probs.device), {}
        
        # Safe log-ratio computation
        log_ratio = log_probs - old_log_probs
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)

        # Broadcast advantages to match sequence length
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        if advantages.shape[1] != log_probs.shape[1]:
            advantages_expanded = advantages.expand_as(log_probs)
        else:
            advantages_expanded = advantages

        # Clipped surrogate objective
        surr1 = ratio * advantages_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_expanded
        
        # Standard PPO loss (unweighted)
        policy_loss_unweighted = -torch.min(surr1, surr2)
        
        # Apply masking
        if masks is not None:
            # Ensure masks match shape
            if masks.shape != policy_loss_unweighted.shape:
                # This might happen if padding was different? Should be handled in recompute.
                # Try to crop/pad
                pass
            
            # Apply confidence weighting if provided (broadcasted)
            if confidence is not None:
                if confidence.dim() == 1:
                    confidence = confidence.unsqueeze(1)
                confidence_expanded = confidence.expand_as(policy_loss_unweighted)
                policy_loss_unweighted = policy_loss_unweighted * confidence_expanded

            # Masked mean
            masked_sum = (policy_loss_unweighted * masks).sum()
            mask_sum = masks.sum()
            policy_loss = masked_sum / (mask_sum + 1e-8)
            
            # Entropy loss (masked)
            entropy_loss = -(entropy * masks).sum() / (mask_sum + 1e-8)
            
            # KL divergence (masked)
            with torch.no_grad():
                # approx_kl = (ratio - 1) - log_ratio  # http://joschu.net/blog/kl-approx.html
                # Simpler approx: 0.5 * (log_ratio)^2
                approx_kl_per_token = 0.5 * (log_ratio ** 2)
                approx_kl = (approx_kl_per_token * masks).sum() / (mask_sum + 1e-8)

        else:
            # Apply confidence weighting if provided
            if confidence is not None:
                if confidence.dim() == 1:
                    confidence = confidence.unsqueeze(1)
                confidence_expanded = confidence.expand_as(policy_loss_unweighted)
                policy_loss = (policy_loss_unweighted * confidence_expanded).mean()
            else:
                policy_loss = policy_loss_unweighted.mean()

            # Entropy bonus (encourage exploration)
            entropy_loss = -entropy.mean()

            # Calculate approximate KL divergence for logging/early stopping
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()

        # Value loss (MSE) - Value is per trajectory, so no masking needed
        value_loss = F.mse_loss(values, returns)

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
            "approx_kl": approx_kl.item(),
            "mean_reward": returns.mean().item(),
            "mean_return": returns.mean().item(),
            "std_return": returns.std().item()
        }

        return total_loss, loss_components
        #     # This is less sensitive to extreme ratios than (ratio - 1) - log_ratio
        #     approx_kl = 0.5 * (log_ratio ** 2).mean()
        #     # per_sample_kl = 0.5 * (log_ratio ** 2)
        #     # per_sample_kl = per_sample_kl.clamp(max=0.5)  # Clamp to disallow a few extreme sample KL values from dominating KL calc
        #     # approx_kl = per_sample_kl.mean()

        # # Total loss
        # total_loss = (
        #     policy_loss +
        #     self.value_coef * value_loss +
        #     self.entropy_coef * entropy_loss
        # )

        # loss_components = {
        #     "policy_loss": policy_loss.item(),
        #     "value_loss": value_loss.item(),
        #     "entropy_loss": entropy_loss.item(),
        #     "total_loss": total_loss.item(),
        #     "approx_kl": approx_kl.item()
        # }

        # return total_loss, loss_components

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
        if self.debug_kl:
            print(
                f"[PPO DEBUG] Log-prob filter: kept {len(all_trajectories)} / {len(raw_trajectories)}"
                f" trajectories (dropped {dropped_count})"
            )

        if not all_trajectories:
            return {"error": "No valid trajectories after filtering"}

        # Sort trajectories by game, agent, and round to ensure temporal order for GAE
        # This is CRITICAL for correct advantage estimation
        all_trajectories.sort(key=lambda t: (t.game_id, t.agent_id, t.round_number))

        rewards = [t.reward for t in all_trajectories]
        
        # Extract confidence scores (default to 1.0 if not present)
        confidences = torch.tensor(
            [getattr(t, 'parsing_confidence', 1.0) for t in all_trajectories], 
            dtype=torch.float32
        )

        pre_recompute_count = len(all_trajectories)

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
                # Updated call: returns 5 values
                batch_log_probs, _, values, _, _ = self._recompute_log_probs_and_values(batch)
                
                # Check for validity of recomputed values
                for j, val in enumerate(values):
                    # batch_log_probs is (B, L)
                    lp = batch_log_probs[j]
                    if torch.isfinite(val) and torch.isfinite(lp).all():
                        initial_values_list.append(val.item())
                        valid_indices.append(i + j)
        
        # --- SECOND FILTERING STEP: Remove trajectories where recomputation failed ---
        if len(valid_indices) < len(all_trajectories):
            print(f"Dropped {len(all_trajectories) - len(valid_indices)} trajectories due to recomputation failure")
            all_trajectories = [all_trajectories[i] for i in valid_indices]
            rewards = [rewards[i] for i in valid_indices]
            confidences = confidences[valid_indices]

        recompute_dropped = pre_recompute_count - len(all_trajectories)
        if self.debug_kl:
            print(
                f"[PPO DEBUG] Recompute filter: kept {len(all_trajectories)} / {pre_recompute_count}"
                f" trajectories (dropped {recompute_dropped})"
            )
            if self._debug_log_diffs:
                diffs = np.array(self._debug_log_diffs)
                finite = diffs[np.isfinite(diffs)]
                nan_count = len(diffs) - len(finite)
                if len(finite) > 0:
                    print(
                        "[PPO DEBUG] log_prob diff stats -> "
                        f"mean {finite.mean():.4f}, std {finite.std():.4f}, "
                        f"min {finite.min():.4f}, max {finite.max():.4f}, n={len(finite)}, nan/inf={nan_count}"
                    )
                else:
                    print(f"[PPO DEBUG] log_prob diff stats -> no finite entries (nan/inf count {nan_count})")

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
        advantages = advantages.to(device)
        returns = returns.to(device)
        confidences = confidences.to(device)

        # --- FILTERING: Keep only Discuss and Vote phases for training ---
        training_indices = []
        for i, t in enumerate(all_trajectories):
            p = (t.phase or "").lower()
            if "discuss" in p or "vote" in p:
                training_indices.append(i)
        
        if not training_indices:
            print("[PPO WARNING] No discuss/vote trajectories found. Skipping update loop.")
            return {"error": "No trainable phases found"}
            
        if self.debug_kl:
            print(f"[PPO DEBUG] Phase filter: kept {len(training_indices)} / {len(all_trajectories)} trajectories (discuss/vote only)")

        # Filter the dataset for the update loop
        train_trajectories = [all_trajectories[i] for i in training_indices]
        train_advantages = advantages[training_indices]
        train_returns = returns[training_indices]
        train_confidences = confidences[training_indices]

        # PPO update loop
        all_metrics = []
        total_mask_tokens = 0.0
        total_trainable_tokens = 0.0
        total_mask_sequences = 0
        total_zero_mask_sequences = 0
        
        # Create indices for shuffling
        dataset_size = len(train_trajectories)
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
                    mini_trajectories = [train_trajectories[i] for i in mini_indices]
                    mini_advantages = train_advantages[mini_indices]
                    mini_returns = train_returns[mini_indices]
                    mini_confidences = train_confidences[mini_indices]
                    
                    # REAL IMPLEMENTATION: Forward pass through model with gradients
                    # Recompute log probs, values, and entropy with current model weights
                    new_log_probs, old_log_probs, new_values, entropy, masks = self._recompute_log_probs_and_values(
                        mini_trajectories,
                        requires_grad=True
                    )

                    batch_total_tokens = float(masks.numel()) if masks.numel() > 0 else 0.0
                    batch_trainable_tokens = float(masks.sum().item()) if batch_total_tokens > 0 else 0.0
                    seq_sums = masks.sum(dim=1)
                    zero_mask_seqs = int(torch.count_nonzero(seq_sums < 1e-6).item()) if masks.shape[0] > 0 else 0
                    trainable_ratio = (batch_trainable_tokens / batch_total_tokens) if batch_total_tokens > 0 else 0.0
                    zero_ratio = (zero_mask_seqs / masks.shape[0]) if masks.shape[0] > 0 else 0.0
                    total_mask_tokens += batch_total_tokens
                    total_trainable_tokens += batch_trainable_tokens
                    total_mask_sequences += masks.shape[0]
                    total_zero_mask_sequences += zero_mask_seqs

                    # Compute PPO loss
                    loss, loss_components = self.compute_ppo_loss(
                        new_log_probs,
                        old_log_probs,
                        mini_advantages,
                        new_values,
                        mini_returns,
                        entropy,
                        masks=masks,
                        confidence=mini_confidences
                    )
                    if self.debug_kl:
                        print(
                            f"[PPO DEBUG] mini-batch approx_kl={loss_components['approx_kl']:.5f}, "
                            f"policy={loss_components['policy_loss']:.4f}, value={loss_components['value_loss']:.4f}, "
                            f"entropy={loss_components['entropy_loss']:.4f}, trainable_tokens={batch_trainable_tokens:.0f}/"
                            f"{batch_total_tokens:.0f} ({trainable_ratio:.1%}), zero-mask seq {zero_mask_seqs}/"
                            f"{masks.shape[0]} ({zero_ratio:.1%})"
                        )
                    
                    # Backward pass (accumulate gradients)
                    # Normalize loss by number of mini-batches to keep scale correct
                    loss = loss / (current_batch_size / mini_batch_size)
                    loss.backward()
                    
                    batch_kls.append(loss_components["approx_kl"])
                    all_metrics.append(loss_components)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Update weights
                self.optimizer.step()
                
                # Check early stopping
                mean_kl = np.mean(batch_kls)
                if self.debug_kl:
                    logical_batch_idx = (start_idx // batch_size) + 1
                    print(
                        f"[PPO DEBUG] Epoch {epoch} logical batch {logical_batch_idx}: mean KL {mean_kl:.5f}"
                    )
                if mean_kl > self.target_kl * 1.5:
                    stop_early = True
                    break
        
        mask_trainable_ratio = (total_trainable_tokens / total_mask_tokens) if total_mask_tokens > 0 else 0.0
        mask_zero_sequence_ratio = (total_zero_mask_sequences / total_mask_sequences) if total_mask_sequences > 0 else 0.0

        if self.debug_kl:
            print(
                f"[PPO DEBUG] Mask summary: trainable tokens {total_trainable_tokens:.0f}/"
                f"{total_mask_tokens:.0f} ({mask_trainable_ratio:.1%}), zero-mask seq {total_zero_mask_sequences}/"
                f"{total_mask_sequences} ({mask_zero_sequence_ratio:.1%})"
            )
            print(
                f"[PPO DEBUG] Trajectories used {len(all_trajectories)} / {len(raw_trajectories)}"
                f" after all filters"
            )

        # Aggregate metrics
        if not all_metrics:
            return {}
            
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        avg_metrics.update({
            "trajectories_total": len(raw_trajectories),
            "trajectories_used": len(all_trajectories),
            "trajectories_dropped_logprob": dropped_count,
            "trajectories_dropped_recompute": recompute_dropped,
            "mask_trainable_ratio": mask_trainable_ratio,
            "mask_zero_sequence_ratio": mask_zero_sequence_ratio
        })
            
        self.training_history.append(avg_metrics)
        self.current_iteration += 1
        
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

    def _create_region_mask(self, text: str, generated_ids: torch.Tensor, phase: str = "unknown") -> torch.Tensor:
        """
        Create a boolean mask for PPO training regions.
        1 = Trainable (Reasoning, Argument)
        0 = Frozen (Scaffolding, Action)
        """
        device = generated_ids.device
        mask = torch.zeros(len(generated_ids), dtype=torch.float32, device=device)
        
        # Normalize phase
        phase = phase.lower() if phase else "unknown"
        
        is_discuss = "discuss" in phase
        is_vote = "vote" in phase
        is_night = "night" in phase

        # If not a trainable phase, return zeros
        if not (is_discuss or is_vote or is_night):
            return mask

        # Find markers
        # Support multiple variations for robustness
        inner_match = re.search(r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|INNER):', text, re.IGNORECASE)
        public_match = re.search(r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', text, re.IGNORECASE)
        action_label_match = re.search(r'ACTION:', text, re.IGNORECASE)
        action_target_match = re.search(r'ACTION:\s*(\S+)', text, re.IGNORECASE)
        
        trainable_spans = []
        
        # Helper to add span
        def add_span(start, end):
            if end > start:
                trainable_spans.append((start, end))
        
        def add_action_target_span():
            if action_target_match and action_target_match.lastindex >= 1:
                start, end = action_target_match.span(1)
                add_span(start, end)

        action_label_pos = action_label_match.start() if action_label_match else len(text)

        # Logic for Discuss phases (No ACTION expected)
        if is_discuss:
            # Region: Inner Thoughts
            if inner_match:
                start = inner_match.end()
                # End at Public Argument if exists, else End of Text
                end = public_match.start() if public_match else len(text)
                add_span(start, end)
            
            # Region: Public Argument
            if public_match:
                start = public_match.end()
                # End at End of Text (ignore ACTION if present, as per instructions)
                end = action_label_pos
                add_span(start, end)

        # Logic for Vote phases (Expects ACTION)
        elif is_vote or is_night:
            # Only train on the specific action target token(s)
            add_action_target_span()

        if not trainable_spans:
            return mask
            
        # Map character spans to tokens
        # We need offsets mapping. Since we don't have it stored, we re-tokenize.
        # This is an approximation but usually accurate for deterministic tokenizers.
        try:
            # Note: generated_ids usually doesn't include BOS, but might include EOS.
            # text is decoded from generated_ids.
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = enc.offset_mapping
            
            # Check if token count matches
            # generated_ids might have EOS/Pad at the end which decode to empty string or are skipped
            # But text was decoded from generated_ids.
            # If len(offsets) != len(generated_ids), it's likely due to special tokens.
            
            # We iterate up to min length
            limit = min(len(offsets), len(generated_ids))
            
            for i in range(limit):
                start, end = offsets[i]
                # Skip zero-length offsets (special tokens)
                if start == end:
                    continue

                # Check if token overlaps any trainable span
                is_trainable = False
                for span_start, span_end in trainable_spans:
                    if start < span_end and end > span_start:
                        is_trainable = True
                        break
                
                if is_trainable:
                    mask[i] = 1.0
                    
        except Exception as e:
            print(f"Error creating region mask: {e}")
            return mask
            
        return mask

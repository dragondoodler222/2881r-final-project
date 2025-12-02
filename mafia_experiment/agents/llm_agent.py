"""
LLM Agent: Agent powered by language model with RL training support
"""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base_agent import BaseAgent, AgentAction
from ..game.roles import Role
from ..utils.prompts import PromptTemplate


class LLMAgent(BaseAgent):
    """
    Agent powered by a language model, supports RL training
    """

    def __init__(
        self,
        agent_id: str,
        role: Optional[Role],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 0.7,
        max_tokens: int = 256
    ):
        super().__init__(agent_id, role)
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens

        # For storing current step's log prob (needed for RL)
        self.last_log_prob = None
        self.last_cot = ""
        self.last_prompt = ""  # Store prompt for PPO recomputation
        self.last_input_ids = None  # Store tokenized input for PPO recomputation
        self.last_generated_ids = None  # Store generated tokens for PPO log prob recomputation
        self.last_parsing_confidence = 1.0  # Store parsing confidence
        self.last_token_log_probs: List[float] = []

    def perceive(self, game_state: Dict, visible_cots: List[Dict]) -> None:
        """
        Process game state and other agents' reasoning

        Args:
            game_state: Current visible game state
            visible_cots: Chain of thoughts from other agents
        """
        # Store in memory
        self.update_memory({
            "type": "perception",
            "game_state": game_state,
            "visible_cots": visible_cots,
            "round": game_state.get("current_round", 0)
        })

    def deliberate(self, action_type: str, game_state: Dict, visible_cots: List[Dict], is_cot_public: bool = False) -> Tuple[str, str]:
        """
        Generate chain of thought and decide on action

        Args:
            action_type: Type of action ("vote", "kill", "save")
            game_state: Current game state
            visible_cots: Visible CoTs from other agents
            is_cot_public: Whether the agent's CoT will be visible to others

        Returns:
            Tuple of (chain_of_thought, action)
        """
        # Build prompt
        if self.role is None:
            raise ValueError(f"Agent {self.agent_id} has no role assigned!")

        prompt = PromptTemplate.build_action_prompt(
            role=self.role,
            action_type=action_type,
            game_state=game_state,
            visible_cots=visible_cots,
            memory=self.memory[-3:],  # Last 3 events
            is_cot_public=is_cot_public
        )

        # Generate response
        cot_and_action, log_prob, input_ids, generated_ids, token_log_probs = self._generate_with_log_prob(prompt)

        # Store prompt, log prob, input_ids, and generated_ids for RL training (PPO recomputation)
        self.last_prompt = prompt
        self.last_log_prob = log_prob
        self.last_cot = cot_and_action
        self.last_input_ids = input_ids
        self.last_generated_ids = generated_ids
        self.last_token_log_probs = token_log_probs

        # Parse CoT and action
        cot, action, confidence = self._parse_response(cot_and_action, game_state, action_type)
        self.last_parsing_confidence = confidence

        return cot, action

    def prepare_act(
        self,
        action_type: str,
        game_state: Dict,
        visible_cots: List[Dict],
        is_cot_public: bool = False
    ) -> str:
        """
        Prepare the prompt for an action (Step 1 of batch generation)

        Args:
            action_type: Type of action
            game_state: Current game state
            visible_cots: Visible CoTs
            is_cot_public: Whether CoT is public

        Returns:
            Prompt string
        """
        # First perceive the environment
        self.perceive(game_state, visible_cots)

        if self.role is None:
            raise ValueError(f"Agent {self.agent_id} has no role assigned!")

        prompt = PromptTemplate.build_action_prompt(
            role=self.role,
            action_type=action_type,
            game_state=game_state,
            visible_cots=visible_cots,
            memory=self.memory[-3:],  # Last 3 events
            is_cot_public=is_cot_public
        )
        return prompt

    def finalize_act(
        self,
        action_type: str,
        prompt: str,
        generated_text: str,
        log_prob: float,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        game_state: Dict,
        token_log_probs: List[float] = None
    ) -> AgentAction:
        """
        Process the generated response and create action (Step 2 of batch generation)

        Args:
            action_type: Type of action
            prompt: The prompt used
            generated_text: The text generated by model
            log_prob: The log probability of generation
            input_ids: Input token IDs
            generated_ids: Generated token IDs
            game_state: Current game state
            token_log_probs: List of log probs per token

        Returns:
            AgentAction object
        """
        # Store prompt, log prob, input_ids, and generated_ids for RL training (PPO recomputation)
        self.last_prompt = prompt
        self.last_log_prob = log_prob
        self.last_cot = generated_text
        self.last_input_ids = input_ids
        self.last_generated_ids = generated_ids
        self.last_token_log_probs = token_log_probs or []

        # Parse CoT and action
        cot, target, confidence = self._parse_response(generated_text, game_state, action_type)
        self.last_parsing_confidence = confidence

        # Create action object
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=action_type,
            target=target,
            reasoning=cot  # Store full CoT as reasoning
        )

        # Store action in memory
        self.update_memory({
            "type": "action",
            "action": action,
            "round": game_state.get("current_round", 0)
        })

        return action

    @staticmethod
    def compute_log_prob_from_scores(
        scores: Tuple[torch.Tensor],
        generated_ids: torch.Tensor,
        eos_token_id: int = 128001,  # Default for Llama-3
        pad_token_id: int = 128001   # Default for Llama-3 (same as EOS)
    ) -> float:
        """
        Compute log probability from generation scores (Static version for batching)

        Args:
            scores: Tuple of logit tensors (one per generated token)
            generated_ids: Generated token IDs
            eos_token_id: EOS token ID to stop accumulation at
            pad_token_id: Pad token ID to skip
            Temperature scaling is already applied inside the generation pipeline

        Returns:
            Sum of log probabilities for the generated sequence
        """
        log_prob_sum = 0.0
        count = 0

        for i, logits in enumerate(scores):
            if i >= len(generated_ids):
                break

            token_id = generated_ids[i].item()
            
            # Skip padding tokens entirely - they weren't actually sampled
            if token_id == pad_token_id:
                continue
                
            # Get log probabilities (scores already include sampling temperature)
            log_probs = torch.log_softmax(logits[0], dim=-1)
            
            # Get log prob of selected token
            token_log_prob = log_probs[token_id].item()
            
            # Handle -inf case (shouldn't happen but be safe)
            if token_log_prob == float('-inf'):
                # Token was filtered by top-k/top-p but somehow selected
                # This indicates padding/EOS confusion - skip it
                continue
            
            log_prob_sum += token_log_prob
            count += 1
            
            # Stop if we hit EOS token (everything after is padding)
            if token_id == eos_token_id:
                break

        # Handle empty sequence case
        if count == 0:
            return -100.0

        # Return sum of log probs (standard for PPO)
        return log_prob_sum

    def act(
        self,
        action_type: str,
        game_state: Dict,
        visible_cots: List[Dict],
        is_cot_public: bool = False
    ) -> AgentAction:
        """
        Take an action in the game

        Args:
            action_type: Type of action ("vote", "kill", "save")
            game_state: Current game state
            visible_cots: Visible CoTs from other agents
            is_cot_public: Whether the agent's CoT will be visible to others

        Returns:
            AgentAction object
        """
        # First perceive the environment
        self.perceive(game_state, visible_cots)

        # Deliberate
        cot, action_target = self.deliberate(action_type, game_state, visible_cots, is_cot_public)

        # Create action object
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=action_type,
            target=action_target,
            reasoning=cot  # Store full CoT as reasoning
        )

        # Store action in memory
        self.update_memory({
            "type": "action",
            "action": action,
            "round": game_state.get("current_round", 0)
        })

        return action

    def _generate_with_log_prob(self, prompt: str) -> Tuple[str, float, torch.Tensor, torch.Tensor, List[float]]:
        """
        Generate text and compute log probability

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, log_prob, input_ids, generated_ids, token_log_probs)
        """
        # If using remote client (model has generate_batch), use it
        if hasattr(self.model, 'generate_batch'):
            results = self.model.generate_batch([prompt], self.max_tokens, self.temperature)
            res = results[0]
            # Assuming remote model also returns token_log_probs, if not we might have an issue.
            # For now, assume it does or handle it.
            token_log_probs = res.get('token_log_probs', [])
            return res['text'], res['log_prob'], res['input_ids'], res['generated_ids'], token_log_probs

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        # Store input_ids for PPO recomputation (on CPU to save GPU memory)
        input_ids = inputs.input_ids.cpu().clone()

        # Generate with temperature sampling
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_k=0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode generated text (only new tokens, not including prompt)
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Store generated_ids for PPO recomputation (on CPU to save GPU memory)
        generated_ids_cpu = generated_ids.cpu().clone()

        # Compute log probability of the generated sequence
        log_prob, token_log_probs = self._compute_log_prob_from_scores(outputs.scores, generated_ids)

        return generated_text, log_prob, input_ids, generated_ids_cpu, token_log_probs

    @staticmethod
    def compute_log_prob_from_scores(
        scores: Tuple[torch.Tensor],
        generated_ids: torch.Tensor,
        eos_token_id: int = None,
        pad_token_id: int = None
    ) -> Tuple[float, List[float]]:
        """
        Compute log probability from generation scores

        Args:
            scores: Tuple of logit tensors (one per generated token)
            generated_ids: Generated token IDs
            eos_token_id: EOS token ID to stop counting
            pad_token_id: Pad token ID to stop counting

        Returns:
            Tuple of (Sum of log probabilities, List of per-token log probabilities)
        """
        log_prob_sum = 0.0
        token_log_probs_list = []
        count = 0

        for i, logits in enumerate(scores):
            if i >= len(generated_ids):
                break
                
            token_id = generated_ids[i].item()
            
            # Stop if we hit padding (EOS is usually included in probability)
            if pad_token_id is not None and token_id == pad_token_id:
                break

            # Get log probabilities
            # Note: logits are already scaled by temperature if generated with do_sample=True
            # But we want the log prob under the sampling distribution, so this is correct.
            log_probs = torch.log_softmax(logits[0], dim=-1)

            token_log_prob = log_probs[token_id].item()
            log_prob_sum += token_log_prob
            token_log_probs_list.append(token_log_prob)
            count += 1
            
            # Stop after processing EOS
            if eos_token_id is not None and token_id == eos_token_id:
                break

        # Handle empty sequence case
        if count == 0:
            return -100.0, []

        # Return sum of log probs (standard for PPO) and list of token log probs
        return log_prob_sum, token_log_probs_list

    def _compute_log_prob_from_scores(
        self,
        scores: Tuple[torch.Tensor],
        generated_ids: torch.Tensor
    ) -> float:
        return LLMAgent.compute_log_prob_from_scores(
            scores, generated_ids, 
            self.tokenizer.eos_token_id, 
            self.tokenizer.pad_token_id
        )

    def _parse_response(
        self,
        response: str,
        game_state: Dict,
        action_type: Optional[str] = None
    ) -> Tuple[str, Optional[str], float]:
        """
        Parse the response into CoT and action using multi-tier strategy

        Args:
            response: Generated text
            game_state: Current game state
            action_type: Type of action that produced this response

        Returns:
            Tuple of (chain_of_thought, target_player, confidence_score)
        """
        import re

        alive_players = game_state.get("alive_players", [])
        other_players = [p for p in alive_players if p != self.agent_id]

        if not other_players:
            return response, None, 1.0

        target = None
        cot = response
        confidence = 0.0
        action_type_norm = (action_type or "").lower()
        expect_action_only = action_type_norm in {"vote", "kill", "save"}

        # TIER 1: Explicit ACTION: format (highest priority)
        # Look for "ACTION: player_X" at the end of response
        action_match = re.search(r'ACTION:\s*(\S+)', response, re.IGNORECASE)
        if action_match:
            candidate = action_match.group(1).strip()
            
            if expect_action_only:
                # Vote / night actions should not emit reasoning
                cot = ""
            else:
                # Extract INTERNAL REASONING if present
                private_match = re.search(
                    r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|PRIVATE THOUGHTS):(.*?)(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):',
                    response,
                    re.DOTALL | re.IGNORECASE
                )
                if private_match:
                    cot = private_match.group(1).strip()
                else:
                    # Fallback: everything before action
                    cot = response[:action_match.start()].strip()

            # Check if it's a valid player
            if candidate in alive_players:
                target = candidate
                confidence = 1.0
            else:
                # Try to find closest match
                for player in alive_players:
                    if player in candidate or candidate in player:
                        target = player
                        confidence = 0.8
                        break
        
        # TIER 2: Structured format without explicit ACTION
        elif re.search(r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|PRIVATE THOUGHTS):', response, re.IGNORECASE) and \
            re.search(r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', response, re.IGNORECASE):
            
            # Extract INTERNAL REASONING
            private_match = re.search(r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|PRIVATE THOUGHTS):(.*?)(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', response, re.DOTALL | re.IGNORECASE)
            if private_match:
                cot = private_match.group(1).strip()
            
            # Try to find target in the whole response
            for player in other_players:
                if player in response:
                    target = player
                    confidence = 0.6
                    break
        
        # TIER 3: Heuristic search (fallback)
        else:
            # Look for player names in the text
            found_players = []
            for player in other_players:
                if player in response:
                    found_players.append(player)
            
            if found_players:
                # Pick the last mentioned player (often the conclusion)
                target = found_players[-1]
                confidence = 0.5

        return cot, target, confidence

    def get_last_log_prob(self) -> Optional[float]:
        """Get log probability of last action (for RL training)"""
        return self.last_log_prob

    def get_last_cot(self) -> str:
        """Get last chain of thought"""
        return self.last_cot

    def get_last_prompt(self) -> str:
        """Get last prompt (for PPO recomputation)"""
        return self.last_prompt

    def get_last_input_ids(self) -> Optional[torch.Tensor]:
        """Get last input_ids (for PPO recomputation)"""
        return self.last_input_ids

    def get_last_generated_ids(self) -> Optional[torch.Tensor]:
        """Get last generated token IDs (for PPO log prob recomputation)"""
        return self.last_generated_ids

    def get_last_parsing_confidence(self) -> float:
        """Get parsing confidence of last action"""
        return self.last_parsing_confidence

    def get_last_temperature(self) -> float:
        """Get temperature used for last action"""
        return self.temperature

    def get_last_token_log_probs(self) -> Optional[List[float]]:
        """Get last token log probs (for masked PPO)"""
        return self.last_token_log_probs

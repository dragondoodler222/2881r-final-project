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

    def deliberate(self, action_type: str, game_state: Dict, visible_cots: List[Dict]) -> Tuple[str, str]:
        """
        Generate chain of thought and decide on action

        Args:
            action_type: Type of action ("vote", "kill", "save")
            game_state: Current game state
            visible_cots: Visible CoTs from other agents

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
            memory=self.memory[-3:]  # Last 3 events
        )

        # Generate response
        cot_and_action, log_prob, input_ids, generated_ids = self._generate_with_log_prob(prompt)

        # Store prompt, log prob, input_ids, and generated_ids for RL training (PPO recomputation)
        self.last_prompt = prompt
        self.last_log_prob = log_prob
        self.last_cot = cot_and_action
        self.last_input_ids = input_ids
        self.last_generated_ids = generated_ids

        # Parse CoT and action
        cot, action, confidence = self._parse_response(cot_and_action, game_state)
        self.last_parsing_confidence = confidence

        return cot, action

    def act(
        self,
        action_type: str,
        game_state: Dict,
        visible_cots: List[Dict]
    ) -> AgentAction:
        """
        Take an action in the game

        Args:
            action_type: Type of action
            game_state: Current game state
            visible_cots: Visible CoTs

        Returns:
            AgentAction object
        """
        # First perceive the environment
        self.perceive(game_state, visible_cots)

        # Deliberate and choose action
        cot, target = self.deliberate(action_type, game_state, visible_cots)

        # Create action
        action = AgentAction(
            agent_id=self.agent_id,
            action_type=action_type,
            target=target,
            reasoning=cot
        )

        # Store action in memory
        self.update_memory({
            "type": "action",
            "action": action,
            "round": game_state.get("current_round", 0)
        })

        return action

    def _generate_with_log_prob(self, prompt: str) -> Tuple[str, float, torch.Tensor, torch.Tensor]:
        """
        Generate text and compute log probability

        Args:
            prompt: Input prompt

        Returns:
            Tuple of (generated_text, log_prob, input_ids, generated_ids)
        """
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
        log_prob = self._compute_log_prob_from_scores(outputs.scores, generated_ids)

        return generated_text, log_prob, input_ids, generated_ids_cpu

    def _compute_log_prob_from_scores(
        self,
        scores: Tuple[torch.Tensor],
        generated_ids: torch.Tensor
    ) -> float:
        """
        Compute log probability from generation scores

        Args:
            scores: Tuple of logit tensors (one per generated token)
            generated_ids: Generated token IDs

        Returns:
            Average log probability per token
        """
        log_prob_sum = 0.0
        count = 0

        for i, logits in enumerate(scores):
            if i >= len(generated_ids):
                break

            # Get log probabilities
            log_probs = torch.log_softmax(logits[0], dim=-1)

            # Get log prob of selected token
            token_id = generated_ids[i].item()
            log_prob_sum += log_probs[token_id].item()
            count += 1

        # Return sum of log probs (standard for PPO)
        return log_prob_sum

    def _parse_response(
        self,
        response: str,
        game_state: Dict
    ) -> Tuple[str, Optional[str], float]:
        """
        Parse the response into CoT and action using multi-tier strategy

        Args:
            response: Generated text
            game_state: Current game state

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

        # TIER 1: Explicit ACTION: format (highest priority)
        # Look for "ACTION: player_X" at the end of response
        action_match = re.search(r'ACTION:\s*(\S+)', response, re.IGNORECASE)
        if action_match:
            candidate = action_match.group(1).strip()
            cot = response[:action_match.start()].strip()

            # Check if it's a valid player
            if candidate in other_players:
                return cot, candidate, 1.0

            # Try partial match (e.g., "player" matches "player_1")
            for player in other_players:
                if candidate in player or player in candidate:
                    return cot, player, 1.0

        # TIER 2: Semantic voting patterns
        # Match common voting phrases with player names
        vote_patterns = [
            r'I\s+(?:will\s+)?vote\s+(?:for\s+|to\s+eliminate\s+)?(\S+)',
            r'(?:my\s+)?vote\s+(?:is\s+|goes\s+to\s+)?(\S+)',
            r'I\s+(?:choose|pick|select)\s+(\S+)',
            r'eliminate\s+(\S+)',
            r'lynch\s+(\S+)',
            r'kill\s+(\S+)',  # For Mafia night actions
            r'target\s+(\S+)',
            r'save\s+(\S+)',  # For Doctor
            r'protect\s+(\S+)'  # For Doctor
        ]

        for pattern in vote_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()

                # Direct match
                if candidate in other_players:
                    # Extract CoT as everything before the vote statement
                    cot = response[:match.start()].strip()
                    return cot, candidate, 0.7

                # Partial match
                for player in other_players:
                    if candidate in player or player in candidate:
                        cot = response[:match.start()].strip()
                        return cot, player, 0.7

        # TIER 3: Last mentioned alive player
        # Find the last occurrence of any alive player's name
        last_position = -1
        last_player = None

        for player in other_players:
            # Find all occurrences of this player
            for match in re.finditer(re.escape(player), response):
                if match.start() > last_position:
                    last_position = match.start()
                    last_player = player

        if last_player is not None:
            # Split CoT at the last player mention (keep it in action part)
            return response[:last_position].strip(), last_player, 0.3

        # TIER 4: Fallback to random selection
        # This ensures the game can always continue
        import random
        target = random.choice(other_players)

        return cot, target, 0.0

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
        """Get parsing confidence of last action (for RL training)"""
        return self.last_parsing_confidence

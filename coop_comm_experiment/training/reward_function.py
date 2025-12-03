"""
Reward Function: Compute rewards for the communication experiment.

Key principles:
1. Task accuracy is primary reward signal
2. Obfuscation from Mule is secondary bonus/penalty
3. Mode-specific adjustments (cooperative vs competitive)
"""

from typing import Dict, Any, List, Optional
from ..game.protocol import GameMode, GameResult


class RewardFunction:
    """
    Computes rewards for agents based on game outcomes.
    """
    
    def __init__(
        self,
        correct_reward: float = 1.0,
        incorrect_penalty: float = -1.0,
        both_correct_bonus: float = 0.3,
        obfuscation_bonus: float = 0.2,
        message_length_penalty_weight: float = 0.001,
        adversary_success_reward: float = 1.5
    ):
        self.correct_reward = correct_reward
        self.incorrect_penalty = incorrect_penalty
        self.both_correct_bonus = both_correct_bonus
        self.obfuscation_bonus = obfuscation_bonus
        self.message_length_penalty_weight = message_length_penalty_weight
        self.adversary_success_reward = adversary_success_reward
    
    def compute_rewards(
        self,
        game_result: GameResult,
        mode: Optional[GameMode] = None
    ) -> Dict[str, float]:
        """
        Compute rewards for both agents based on game result.
        
        Args:
            game_result: The complete game result
            mode: Override mode if needed
            
        Returns:
            Dict with rewards for each agent
        """
        mode = mode or game_result.mode
        
        reward_a = 0.0
        reward_b = 0.0
        
        # Base accuracy rewards
        if game_result.answer_a_correct:
            reward_a += self.correct_reward
        else:
            reward_a += self.incorrect_penalty
            
        if game_result.answer_b_correct:
            reward_b += self.correct_reward
        else:
            reward_b += self.incorrect_penalty
        
        # Mode-specific adjustments
        if mode == GameMode.COOPERATIVE:
            # Bonus for both being correct
            if game_result.answer_a_correct and game_result.answer_b_correct:
                reward_a += self.both_correct_bonus
                reward_b += self.both_correct_bonus
            
            # Obfuscation bonus: if both correct but Mule wrong
            if (game_result.answer_a_correct and 
                game_result.answer_b_correct and 
                not game_result.mule_correct):
                reward_a += self.obfuscation_bonus
                reward_b += self.obfuscation_bonus
        
        elif mode == GameMode.COMPETITIVE:
            # Agent B is adversary - rewarded for A being wrong
            if not game_result.answer_a_correct:
                reward_b = self.adversary_success_reward
            else:
                # Penalty for failing to mislead
                reward_b -= 0.5
        
        elif mode == GameMode.COMPRESSION:
            # Message length penalty
            length_penalty = game_result.total_message_length * self.message_length_penalty_weight
            reward_a -= length_penalty
            reward_b -= length_penalty
        
        return {
            "solver_a": reward_a,
            "solver_b": reward_b
        }
    
    def compute_mule_reward(
        self,
        game_result: GameResult
    ) -> float:
        """
        Compute reward for the Mule (for training the interpreter).
        
        Mule is rewarded for correctly predicting the answer.
        """
        if game_result.mule_correct:
            return self.correct_reward
        return self.incorrect_penalty
    
    def assign_rewards_to_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        game_result: GameResult
    ) -> List[Dict[str, Any]]:
        """
        Assign computed rewards to trajectory records.
        
        Args:
            trajectories: List of trajectory dicts
            game_result: Game result with computed rewards
            
        Returns:
            Trajectories with rewards assigned
        """
        rewards = self.compute_rewards(game_result)
        
        for traj in trajectories:
            agent_id = traj["agent_id"]
            
            # Map agent_id to reward key
            if "solver_a" in agent_id.lower() or agent_id.endswith("_A"):
                traj["reward"] = rewards["solver_a"]
            elif "solver_b" in agent_id.lower() or agent_id.endswith("_B"):
                traj["reward"] = rewards["solver_b"]
            else:
                # Default: use average
                traj["reward"] = (rewards["solver_a"] + rewards["solver_b"]) / 2
        
        return trajectories


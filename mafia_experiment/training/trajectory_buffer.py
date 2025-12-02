"""
Trajectory Buffer: Store and manage game trajectories for training
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import random


@dataclass
class Trajectory:
    """
    Single step in a game trajectory
    """
    agent_id: str
    game_id: str
    round_number: int
    phase: str  # "night" or "day"
    state: Dict[str, Any]  # Visible game state
    action: str  # Action taken
    log_prob: float  # Log probability of the action
    reward: float = 0.0  # Reward (assigned later)
    cot: str = ""  # Chain of thought
    visible_cots: List[Dict] = field(default_factory=list)  # CoTs from other agents
    prompt: str = ""  # Full prompt used for this action (needed for PPO recomputation)
    input_ids: Optional[torch.Tensor] = None  # Tokenized input (needed for PPO recomputation)
    generated_ids: Optional[torch.Tensor] = None  # Generated token IDs (for PPO log prob recomputation)
    token_log_probs: Optional[List[float]] = None  # Log probs per token (needed for masked PPO)
    parsing_confidence: float = 1.0  # Confidence in parsing the action (1.0=explicit, 0.0=random)
    temperature: float = 1.0  # Temperature used for sampling

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "game_id": self.game_id,
            "round_number": self.round_number,
            "phase": self.phase,
            "state": self.state,
            "action": self.action,
            "log_prob": self.log_prob,
            "reward": self.reward,
            "cot": self.cot,
            "num_visible_cots": len(self.visible_cots),
            "parsing_confidence": self.parsing_confidence
        }


class TrajectoryBuffer:
    """
    Buffer to store game trajectories for batch training
    """

    def __init__(self, max_size: Optional[int] = None):
        self.trajectories: List[Trajectory] = []
        self.max_size = max_size

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add a trajectory step to the buffer

        Args:
            trajectory: Trajectory object to add
        """
        self.trajectories.append(trajectory)

        # If buffer exceeds max size, remove oldest trajectories
        if self.max_size and len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size:]

    def add_trajectories(self, trajectories: List[Trajectory]) -> None:
        """Add multiple trajectories"""
        for traj in trajectories:
            self.add_trajectory(traj)

    def sample_batch(self, batch_size: Optional[int] = None) -> List[Trajectory]:
        """
        Sample a batch of trajectories

        Args:
            batch_size: Number of trajectories to sample (None = all)

        Returns:
            List of sampled trajectories
        """
        if batch_size is None or batch_size >= len(self.trajectories):
            return self.trajectories.copy()

        return random.sample(self.trajectories, batch_size)

    def get_all(self) -> List[Trajectory]:
        """Get all trajectories"""
        return self.trajectories.copy()

    def get_by_agent(self, agent_id: str) -> List[Trajectory]:
        """Get trajectories for a specific agent"""
        return [t for t in self.trajectories if t.agent_id == agent_id]

    def get_by_game(self, game_id: str) -> List[Trajectory]:
        """Get trajectories for a specific game"""
        return [t for t in self.trajectories if t.game_id == game_id]

    def clear(self) -> None:
        """Clear all trajectories"""
        self.trajectories = []

    def __len__(self) -> int:
        return len(self.trajectories)

    def compute_returns(
        self,
        gamma: float = 0.99,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute discounted returns for all trajectories

        Args:
            gamma: Discount factor
            normalize: Whether to normalize returns

        Returns:
            Tensor of returns
        """
        # Group trajectories by game and agent
        game_agent_trajs = {}
        for traj in self.trajectories:
            key = (traj.game_id, traj.agent_id)
            if key not in game_agent_trajs:
                game_agent_trajs[key] = []
            game_agent_trajs[key].append(traj)

        # Compute returns for each episode
        all_returns = []
        for trajs in game_agent_trajs.values():
            # Sort by round number
            trajs = sorted(trajs, key=lambda t: t.round_number)

            # Compute discounted returns
            returns = []
            G = 0
            for traj in reversed(trajs):
                G = traj.reward + gamma * G
                returns.insert(0, G)

            all_returns.extend(returns)

        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)

        # Normalize returns
        if normalize and len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )

        return returns_tensor

    def get_log_probs(self) -> torch.Tensor:
        """Get log probabilities as tensor"""
        log_probs = [t.log_prob for t in self.trajectories]
        return torch.tensor(log_probs, dtype=torch.float32)

    def statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.trajectories:
            return {
                "size": 0,
                "num_games": 0,
                "num_agents": 0,
                "avg_reward": 0.0
            }

        games = set(t.game_id for t in self.trajectories)
        agents = set(t.agent_id for t in self.trajectories)
        rewards = [t.reward for t in self.trajectories]

        return {
            "size": len(self.trajectories),
            "num_games": len(games),
            "num_agents": len(agents),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0
        }

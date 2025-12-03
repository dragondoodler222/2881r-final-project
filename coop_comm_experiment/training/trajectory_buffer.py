"""
Trajectory Buffer: Store game trajectories for RL training.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch
import random


@dataclass
class Trajectory:
    """Single step in a game trajectory."""
    agent_id: str
    game_id: str
    round_number: int
    phase: str  # "message" or "answer"
    action: str
    log_prob: float
    reward: float = 0.0
    prompt: str = ""
    raw_response: str = ""
    input_ids: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    token_log_probs: Optional[List[float]] = None
    is_adversary: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "game_id": self.game_id,
            "round_number": self.round_number,
            "phase": self.phase,
            "action": self.action,
            "log_prob": self.log_prob,
            "reward": self.reward,
            "is_adversary": self.is_adversary
        }


class TrajectoryBuffer:
    """Buffer to store and sample game trajectories."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.trajectories: List[Trajectory] = []
        self.max_size = max_size
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the buffer."""
        self.trajectories.append(trajectory)
        
        if self.max_size and len(self.trajectories) > self.max_size:
            self.trajectories = self.trajectories[-self.max_size:]
    
    def add_trajectories(self, trajectories: List[Dict[str, Any]]) -> None:
        """Add multiple trajectories from dicts."""
        for traj_dict in trajectories:
            traj = Trajectory(
                agent_id=traj_dict["agent_id"],
                game_id=traj_dict["game_id"],
                round_number=traj_dict["round_number"],
                phase=traj_dict["phase"],
                action=traj_dict["action"],
                log_prob=traj_dict["log_prob"],
                reward=traj_dict.get("reward", 0.0),
                prompt=traj_dict.get("prompt", ""),
                raw_response=traj_dict.get("raw_response", ""),
                input_ids=traj_dict.get("input_ids"),
                generated_ids=traj_dict.get("generated_ids"),
                token_log_probs=traj_dict.get("token_log_probs"),
                is_adversary=traj_dict.get("is_adversary", False)
            )
            self.add_trajectory(traj)
    
    def sample_batch(self, batch_size: Optional[int] = None) -> List[Trajectory]:
        """Sample a batch of trajectories."""
        if batch_size is None or batch_size >= len(self.trajectories):
            return self.trajectories.copy()
        return random.sample(self.trajectories, batch_size)
    
    def get_all(self) -> List[Trajectory]:
        """Get all trajectories."""
        return self.trajectories.copy()
    
    def get_by_agent(self, agent_id: str) -> List[Trajectory]:
        """Get trajectories for specific agent."""
        return [t for t in self.trajectories if t.agent_id == agent_id]
    
    def clear(self) -> None:
        """Clear all trajectories."""
        self.trajectories = []
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if not self.trajectories:
            return {"size": 0, "num_games": 0, "avg_reward": 0.0}
        
        games = set(t.game_id for t in self.trajectories)
        rewards = [t.reward for t in self.trajectories]
        
        return {
            "size": len(self.trajectories),
            "num_games": len(games),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards)
        }


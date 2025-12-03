"""
Configuration management for experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

from ..game.protocol import GameMode
from ..tasks.task_generator import TaskType


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment run."""
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    
    # Training settings
    learning_rate: float = 1e-5
    num_iterations: int = 100
    games_per_iteration: int = 10
    batch_size: int = 16
    
    # Game settings
    game_mode: GameMode = GameMode.COOPERATIVE
    num_message_rounds: int = 2
    max_message_length: int = 200
    is_cot_public: bool = False
    
    # Task settings
    task_types: List[TaskType] = field(default_factory=lambda: list(TaskType))
    task_difficulty: int = 1
    balanced_tasks: bool = True
    
    # Evaluation settings
    eval_games_per_checkpoint: int = 20
    checkpoint_interval: int = 10
    
    # Paths
    checkpoint_dir: str = "checkpoints/coop_comm"
    log_dir: str = "logs/coop_comm"
    
    # Seed
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "num_iterations": self.num_iterations,
            "games_per_iteration": self.games_per_iteration,
            "batch_size": self.batch_size,
            "game_mode": self.game_mode.value,
            "num_message_rounds": self.num_message_rounds,
            "max_message_length": self.max_message_length,
            "is_cot_public": self.is_cot_public,
            "task_types": [t.value for t in self.task_types],
            "task_difficulty": self.task_difficulty,
            "eval_games_per_checkpoint": self.eval_games_per_checkpoint,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "seed": self.seed
        }


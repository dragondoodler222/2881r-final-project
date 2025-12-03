"""
Cooperative/Competitive Communication Experiment

Two solver agents (A and B) receive partial information about a shared task
and must exchange messages before submitting answers. A third Mule model
attempts to decode/interpret their messages.

Research Questions:
1. Does interpretability loss emerge naturally from strategic communication?
2. How does competition vs cooperation affect message clarity?
3. Can compression pressure induce obfuscation even without adversarial incentives?
"""

from .game.protocol import CommunicationProtocol, GameMode
from .tasks.task_generator import TaskGenerator
from .agents.solver_agent import SolverAgent
from .agents.mule_agent import MuleAgent
from .training.trainer import CoopCommTrainer

__all__ = [
    "CommunicationProtocol",
    "GameMode", 
    "TaskGenerator",
    "SolverAgent",
    "MuleAgent",
    "CoopCommTrainer"
]


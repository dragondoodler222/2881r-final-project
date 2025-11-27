"""Agent implementations for the Mafia game"""

from .base_agent import BaseAgent, AgentAction
from .llm_agent import LLMAgent

__all__ = ["BaseAgent", "AgentAction", "LLMAgent"]

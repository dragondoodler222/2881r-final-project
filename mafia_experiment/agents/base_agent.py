"""
Base agent class for Mafia game
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from ..game.roles import Role


@dataclass
class AgentAction:
    """Represents an action taken by an agent"""
    agent_id: str
    action_type: str  # "vote", "kill", "save", "discuss"
    target: Optional[str] = None  # Target player ID
    reasoning: str = ""  # Public reasoning for the action


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Mafia game
    """

    def __init__(self, agent_id: str, role: Optional[Role] = None):
        self.agent_id = agent_id
        self.role = role  # May be None initially, assigned by game engine
        self.is_alive = True
        self.memory: List[Dict[str, Any]] = []  # Store game events and observations

    @abstractmethod
    def perceive(self, game_state: Dict, visible_cots: List[Dict]) -> None:
        """
        Process the current game state and other agents' chain of thought

        Args:
            game_state: Current visible game state for this agent
            visible_cots: List of CoT entries visible to this agent
        """
        pass

    @abstractmethod
    def deliberate(self, action_type: str) -> Tuple[str, str]:
        """
        Generate chain of thought and decide on an action

        Args:
            action_type: Type of action to take ("vote", "kill", "save", "discuss")

        Returns:
            Tuple of (chain_of_thought, action_decision)
        """
        pass

    @abstractmethod
    def act(self, action_type: str, game_state: Dict, visible_cots: List[Dict]) -> AgentAction:
        """
        Take an action in the game

        Args:
            action_type: Type of action to take
            game_state: Current visible game state
            visible_cots: Visible chain of thoughts from other agents

        Returns:
            AgentAction object with the chosen action
        """
        pass

    def update_memory(self, event: Dict[str, Any]) -> None:
        """Add an event to the agent's memory"""
        self.memory.append(event)

    def get_role_description(self) -> str:
        """Get description of the agent's role"""
        if self.role is None:
            return "Role not yet assigned"
        return self.role.get_description()

    def is_mafia_member(self) -> bool:
        """Check if this agent is Mafia"""
        if self.role is None:
            return False
        return self.role.is_mafia

    def has_night_action(self) -> bool:
        """Check if this agent can act during night phase"""
        if self.role is None:
            return False
        return self.role.has_night_action

    def reset(self) -> None:
        """Reset agent state for a new game"""
        self.role = None
        self.is_alive = True
        self.memory = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role}, alive={self.is_alive})"

"""
Game state management for Mafia
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from .roles import Role, RoleType
from .phases import PhaseResult, PhaseType


@dataclass
class GameState:
    """
    Represents the complete state of a Mafia game
    """
    game_id: str
    players: List[str]  # List of player IDs
    roles: Dict[str, Role]  # player_id -> Role (hidden information)
    alive_players: Set[str] = field(default_factory=set)
    dead_players: Set[str] = field(default_factory=set)
    current_round: int = 0
    current_phase: Optional[PhaseType] = None
    phase_history: List[PhaseResult] = field(default_factory=list)
    public_events: List[str] = field(default_factory=list)  # Events visible to all players

    def __post_init__(self):
        """Initialize alive players"""
        if not self.alive_players:
            self.alive_players = set(self.players)

    def get_visible_state(self, agent_id: str) -> Dict:
        """
        Get the game state visible to a specific agent
        Agents can see:
        - Who is alive/dead
        - Public events (eliminations, etc.)
        - Their own role
        - If they're Mafia, they can see other Mafia members
        """
        agent_role = self.roles.get(agent_id)

        visible_state = {
            "game_id": self.game_id,
            "your_id": agent_id,
            "self_player_id": agent_id,
            "your_role": str(agent_role.role_type) if agent_role else None,
            "alive_players": sorted(list(self.alive_players)),
            "dead_players": sorted(list(self.dead_players)),
            "current_round": self.current_round,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "public_events": self.public_events.copy(),
            "num_players_alive": len(self.alive_players)
        }

        # Mafia can see other Mafia members
        if agent_role and agent_role.is_mafia:
            mafia_members = [
                pid for pid, role in self.roles.items()
                if role.is_mafia and pid in self.alive_players
            ]
            visible_state["mafia_team"] = sorted(mafia_members)

        return visible_state

    def is_alive(self, agent_id: str) -> bool:
        """Check if an agent is alive"""
        return agent_id in self.alive_players

    def kill_player(self, agent_id: str, reason: str = "eliminated"):
        """Remove a player from the game"""
        if agent_id in self.alive_players:
            self.alive_players.remove(agent_id)
            self.dead_players.add(agent_id)
            role = self.roles.get(agent_id)
            role_str = str(role.role_type) if role else "Unknown"
            self.public_events.append(
                f"Round {self.current_round}: {agent_id} was {reason}. They were a {role_str}."
            )

    def get_mafia_members(self) -> List[str]:
        """Get all Mafia members (alive or dead)"""
        return [pid for pid, role in self.roles.items() if role.is_mafia]

    def get_alive_mafia(self) -> List[str]:
        """Get alive Mafia members"""
        return [pid for pid in self.get_mafia_members() if pid in self.alive_players]

    def get_alive_village(self) -> List[str]:
        """Get alive village team members"""
        return [pid for pid, role in self.roles.items()
                if role.is_village and pid in self.alive_players]

    def check_win_condition(self) -> Optional[str]:
        """
        Check if the game has ended
        Returns: "mafia", "village", or None
        """
        alive_mafia = len(self.get_alive_mafia())
        alive_village = len(self.get_alive_village())

        # Mafia wins if they equal or outnumber village
        if alive_mafia >= alive_village:
            return "mafia"

        # Village wins if all Mafia are eliminated
        if alive_mafia == 0:
            return "village"

        return None

    def add_public_event(self, event: str):
        """Add an event visible to all players"""
        self.public_events.append(f"Round {self.current_round}: {event}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization (without revealing hidden info)"""
        return {
            "game_id": self.game_id,
            "players": self.players,
            "alive_players": sorted(list(self.alive_players)),
            "dead_players": sorted(list(self.dead_players)),
            "current_round": self.current_round,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "public_events": self.public_events,
            "winner": self.check_win_condition()
        }

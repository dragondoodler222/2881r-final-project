"""
Phase definitions and logic for Mafia game
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


class PhaseType(Enum):
    """Types of game phases"""
    NIGHT = "night"
    DAY = "day"


@dataclass
class Phase:
    """
    Represents a phase in the game
    """
    phase_type: PhaseType
    round_number: int
    time_limit: Optional[int] = None  # in seconds, None for unlimited

    def __str__(self) -> str:
        return f"Round {self.round_number} - {self.phase_type.value.title()}"


@dataclass
class PhaseResult:
    """
    Result of a completed phase
    """
    phase: Phase
    actions: List[Dict[str, Any]]  # List of actions taken during this phase
    eliminated_player: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "phase_type": self.phase.phase_type.value,
            "round_number": self.phase.round_number,
            "actions": self.actions,
            "eliminated_player": self.eliminated_player
        }


@dataclass
class NightPhaseResult(PhaseResult):
    """
    Result specific to night phase
    """
    mafia_target: Optional[str] = None
    doctor_save: Optional[str] = None
    killed_player: Optional[str] = None  # After resolving Mafia kill and Doctor save

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = super().to_dict()
        result.update({
            "mafia_target": self.mafia_target,
            "doctor_save": self.doctor_save,
            "killed_player": self.killed_player
        })
        return result


@dataclass
class DayPhaseResult(PhaseResult):
    """
    Result specific to day phase (discussion + voting)
    """
    lynched_player: str = ""  # ALWAYS someone (no ties allowed) - must be set by game engine
    discussion_round_1: List[Dict[str, str]] = field(default_factory=list)
    discussion_round_2: List[Dict[str, str]] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)
    vote_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = super().to_dict()
        result.update({
            "discussion_round_1": self.discussion_round_1,
            "discussion_round_2": self.discussion_round_2,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
            "lynched_player": self.lynched_player
        })
        return result

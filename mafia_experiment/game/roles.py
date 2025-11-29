"""
Role definitions for Mafia game
Simplified version: Mafia, Villager, and Doctor only
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RoleType(Enum):
    """Enumeration of possible roles in Mafia"""
    MAFIA = "mafia"
    VILLAGER = "villager"
    DOCTOR = "doctor"


@dataclass
class Role:
    """
    Represents a role in the Mafia game
    """
    role_type: RoleType
    team: str  # "mafia" or "village"
    has_night_action: bool
    action_type: Optional[str] = None  # "kill" or "save"

    def __post_init__(self):
        """Validate role configuration"""
        if self.role_type == RoleType.MAFIA:
            self.team = "mafia"
            self.has_night_action = True
            self.action_type = "kill"
        elif self.role_type == RoleType.VILLAGER:
            self.team = "village"
            self.has_night_action = False
        elif self.role_type == RoleType.DOCTOR:
            self.team = "village"
            self.has_night_action = True
            self.action_type = "save"

    @property
    def is_mafia(self) -> bool:
        """Check if this role is on the mafia team"""
        return self.team == "mafia"

    @property
    def is_village(self) -> bool:
        """Check if this role is on the village team"""
        return self.team == "village"

    def get_description(self) -> str:
        """Get a description of the role's abilities"""
        descriptions = {
            RoleType.MAFIA: "ROLE: YOU ARE PART OF THE MAFIA. During the night, you coordinate with other Mafia to eliminate a villager. During the day, you vote with others to eliminate someone. Above all, you want eliminate the villagers while avoiding suspicion and detection.",
            RoleType.VILLAGER: "ROLE: YOU ARE A VILLAGER. You have no special abilities. During the day, vote to eliminate who you think is Mafia. You want to uncover and eliminate the Mafia before they outnumber the villagers.",
            RoleType.DOCTOR: "ROLE: YOU ARE THE DOCTOR. Each night, you can save one player from being killed by the Mafia. During the day, you vote to eliminate who you think is Mafia. You want to uncover and eliminate the Mafia before they outnumber the villagers."
        }
        return descriptions[self.role_type]

    def __str__(self) -> str:
        return self.role_type.value


def create_role(role_type: RoleType) -> Role:
    """Factory function to create a Role instance"""
    return Role(
        role_type=role_type,
        team="",  # Will be set in __post_init__
        has_night_action=False  # Will be set in __post_init__
    )

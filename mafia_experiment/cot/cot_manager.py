"""
CoT Manager: Handle chain-of-thought visibility and logging
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class VisibilityMode(Enum):
    """CoT visibility modes"""
    PUBLIC = "public"  # All CoTs visible to all agents
    PRIVATE = "private"  # No CoTs visible
    ROLE_BASED = "role_based"  # Mafia can see each other's CoTs


@dataclass
class CoTEntry:
    """Single chain-of-thought entry"""
    agent_id: str
    round_number: int
    phase: str
    cot_text: str
    action_type: str
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    public_argument: str = ""  # New field for public argument

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "round_number": self.round_number,
            "phase": self.phase,
            "cot_text": self.cot_text,
            "public_argument": self.public_argument,
            "action_type": self.action_type,
            "timestamp": self.timestamp
        }


class CoTManager:
    """
    Manages chain-of-thought visibility and logging
    """

    # Night actions should remain private to the acting agent
    NIGHT_ONLY_ACTIONS = {"kill", "save"}

    def __init__(self, visibility_mode: VisibilityMode = VisibilityMode.PUBLIC):
        self.visibility_mode = visibility_mode
        self.cot_log: List[CoTEntry] = []
        self.role_assignments: Dict[str, str] = {}  # agent_id -> role

    def _should_hide_entry(self, entry: CoTEntry, requesting_agent_id: str) -> bool:
        """Return True if this entry should be hidden from the requesting agent."""
        if entry.agent_id == requesting_agent_id:
            return False

        phase = (entry.phase or "").lower()
        action = (entry.action_type or "").lower()

        if phase.startswith("night"):
            return True

        if action in self.NIGHT_ONLY_ACTIONS:
            return True

        return False

    def set_role_assignments(self, roles: Dict[str, Any]) -> None:
        """
        Set role assignments for role-based visibility

        Args:
            roles: Dictionary mapping agent_id to Role objects
        """
        self.role_assignments = {
            agent_id: str(role.role_type)
            for agent_id, role in roles.items()
        }

    def record_cot(
        self,
        agent_id: str,
        cot_text: str,
        round_number: int,
        phase: str,
        action_type: str,
        public_argument: str = ""
    ) -> CoTEntry:
        """
        Record a chain-of-thought entry

        Args:
            agent_id: Agent identifier
            cot_text: The chain of thought text
            round_number: Game round number
            phase: "night" or "day"
            action_type: Type of action
            public_argument: The public argument (if any)

        Returns:
            Created CoTEntry
        """
        entry = CoTEntry(
            agent_id=agent_id,
            round_number=round_number,
            phase=phase,
            cot_text=cot_text,
            action_type=action_type,
            public_argument=public_argument
        )

        self.cot_log.append(entry)
        return entry

    def get_visible_cots(
        self,
        requesting_agent_id: str,
        current_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get CoTs visible to a specific agent

        Args:
            requesting_agent_id: Agent requesting CoTs
            current_round: Optional round number to filter by

        Returns:
            List of visible CoT dictionaries
        """
        if self.visibility_mode == VisibilityMode.PRIVATE:
            # No CoTs visible in private mode
            return []

        # Filter by round if specified
        cots = self.cot_log
        if current_round is not None:
            cots = [c for c in cots if c.round_number == current_round]

        if self.visibility_mode == VisibilityMode.PUBLIC:
            # All CoTs visible except agent's own
            # In PUBLIC mode, we show BOTH private thoughts and public argument
            visible = []
            for c in cots:
                if c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                    d = c.to_dict()
                    # Combine private and public for display
                    if d.get("public_argument"):
                        d["display_text"] = f"PRIVATE: {d['cot_text']}\nPUBLIC: {d['public_argument']}"
                    else:
                        d["display_text"] = d["cot_text"]
                    visible.append(d)
            return visible

        elif self.visibility_mode == VisibilityMode.PRIVATE:
            # In PRIVATE mode, we show ONLY public argument
            visible = []
            for c in cots:
                if c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                    d = c.to_dict()
                    # Only show public argument if available
                    if d.get("public_argument"):
                        d["display_text"] = f"PUBLIC: {d['public_argument']}"
                        # Hide private thought
                        d["cot_text"] = "[HIDDEN]"
                        visible.append(d)
                    # If no public argument (e.g. night action), show nothing or generic
            return visible

        elif self.visibility_mode == VisibilityMode.ROLE_BASED:
            # Mafia can see other Mafia CoTs
            requesting_role = self.role_assignments.get(requesting_agent_id, "")

            if requesting_role == "mafia":
                # Mafia can see other Mafia CoTs (full visibility)
                visible = []
                for c in cots:
                    if (
                        self.role_assignments.get(c.agent_id, "") == "mafia"
                        and c.agent_id != requesting_agent_id
                        and not self._should_hide_entry(c, requesting_agent_id)
                    ):
                        d = c.to_dict()
                        if d.get("public_argument"):
                            d["display_text"] = f"PRIVATE: {d['cot_text']}\nPUBLIC: {d['public_argument']}"
                        else:
                            d["display_text"] = d["cot_text"]
                        visible.append(d)
                    elif c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                        # See non-Mafia public arguments only
                        d = c.to_dict()
                        if d.get("public_argument"):
                            d["display_text"] = f"PUBLIC: {d['public_argument']}"
                            d["cot_text"] = "[HIDDEN]"
                            visible.append(d)
                return visible
            else:
                # Other roles see only public arguments
                visible = []
                for c in cots:
                    if c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                        d = c.to_dict()
                        if d.get("public_argument"):
                            d["display_text"] = f"PUBLIC: {d['public_argument']}"
                            d["cot_text"] = "[HIDDEN]"
                            visible.append(d)
                return visible

        return []

    def get_all_cots(
        self,
        round_number: Optional[int] = None,
        phase: Optional[str] = None
    ) -> List[CoTEntry]:
        """
        Get all CoTs (for analysis)

        Args:
            round_number: Filter by round
            phase: Filter by phase

        Returns:
            List of CoTEntry objects
        """
        cots = self.cot_log

        if round_number is not None:
            cots = [c for c in cots if c.round_number == round_number]

        if phase is not None:
            cots = [c for c in cots if c.phase == phase]

        return cots

    def clear_log(self) -> None:
        """Clear all recorded CoTs"""
        self.cot_log = []

    def export_log(self) -> List[Dict[str, Any]]:
        """Export all CoTs as dictionaries"""
        return [cot.to_dict() for cot in self.cot_log]

    def __len__(self) -> int:
        return len(self.cot_log)

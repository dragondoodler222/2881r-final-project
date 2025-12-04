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

    @staticmethod
    def _strip_label(text: str) -> str:
        """Remove leading labels like 'INTERNAL REASONING:', 'PUBLIC ARGUMENT:', etc.
        
        Only strips complete label phrases at the start of text to avoid
        accidentally removing content that happens to start with similar words.
        """
        import re
        # Strip specific leading label patterns (must be complete phrases)
        text = re.sub(r'^\s*INTERNAL REASONING\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*INNER THOUGHTS\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*PUBLIC ARGUMENT\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*PUBLIC STATEMENT\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*ACTION\s*:\s*', '', text, flags=re.IGNORECASE)
        return text.strip()

    @staticmethod
    def _extract_internal_only(text: str) -> str:
        """Extract only the INTERNAL REASONING portion, excluding PUBLIC ARGUMENT if present."""
        import re
        # First strip leading label
        text = CoTManager._strip_label(text)
        
        # If there's a PUBLIC ARGUMENT section embedded, extract only the part before it
        public_match = re.search(
            r'\s*(?:PUBLIC ARGUMENT|PUBLIC STATEMENT)\s*:',
            text, re.IGNORECASE
        )
        if public_match:
            # Return only the text before PUBLIC ARGUMENT
            return text[:public_match.start()].strip()
        
        # Also check for ACTION: which might follow internal reasoning
        action_match = re.search(r'\s*ACTION\s*:', text, re.IGNORECASE)
        if action_match:
            return text[:action_match.start()].strip()
        
        return text.strip()

    @staticmethod
    def _build_display_sections(
        entry_dict: Dict[str, Any],
        include_private: bool,
        include_public: bool
    ) -> Dict[str, Any]:
        """Return structured display data for a CoT entry."""
        agent_label = (entry_dict.get("agent_id") or "").strip()
        action_type = (entry_dict.get("action_type") or "").lower()
        phase = (entry_dict.get("phase") or "").lower()

        # Extract only the internal reasoning portion (excluding any embedded PUBLIC ARGUMENT)
        cot_text = (entry_dict.get("cot_text") or "").strip()
        raw_private = CoTManager._extract_internal_only(cot_text)
        
        # For public argument, use the dedicated field if available, otherwise try to extract from cot_text
        public_arg_field = (entry_dict.get("public_argument") or "").strip()
        if public_arg_field:
            raw_public = CoTManager._strip_label(public_arg_field)
        else:
            # Try to extract PUBLIC ARGUMENT from cot_text if not separately provided
            import re
            public_match = re.search(
                r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT)\s*:\s*(.*?)(?=\s*ACTION:|$)',
                cot_text, re.DOTALL | re.IGNORECASE
            )
            raw_public = public_match.group(1).strip() if public_match else ""

        combined_sections: List[str] = []
        private_formatted = ""
        public_formatted = ""

        # Never expose night-only actions regardless of visibility
        if action_type in CoTManager.NIGHT_ONLY_ACTIONS or phase.startswith("night"):
            include_private = False
            include_public = False
            raw_private = ""
            raw_public = ""

        if include_private and raw_private:
            if action_type in {"vote", "kill", "save"}:
                label = "ACTION OUTPUT"
            else:
                label = "INTERNAL REASONING"
            prefix = f"[{agent_label}] " if agent_label else ""
            private_formatted = f"{label}: {prefix}{raw_private}"
            combined_sections.append(private_formatted)
        else:
            raw_private = ""

        if include_public and raw_public:
            if action_type.startswith("discuss"):
                public_label = "PUBLIC ARGUMENT"
            else:
                public_label = "PUBLIC STATEMENT"
            prefix = f"[{agent_label}] " if agent_label else ""
            public_formatted = f"{public_label}: {prefix}{raw_public}"
            combined_sections.append(public_formatted)
        else:
            raw_public = ""

        combined = "\n".join([s for s in combined_sections if s]).strip()

        return {
            "agent_id": agent_label,
            "action_type": action_type,
            "private_raw": raw_private,
            "public_raw": raw_public,
            "private_formatted": private_formatted,
            "public_formatted": public_formatted,
            "combined": combined
        }

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
                    sections = self._build_display_sections(d, include_private=True, include_public=True)
                    d["display_sections"] = sections
                    d["display_text"] = sections["combined"] or d.get("cot_text", "")
                    visible.append(d)
            return visible

        elif self.visibility_mode == VisibilityMode.PRIVATE:
            # In PRIVATE mode, we show ONLY public argument
            visible = []
            for c in cots:
                if c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                    d = c.to_dict()
                    sections = self._build_display_sections(d, include_private=False, include_public=True)
                    if sections["combined"]:
                        d["display_sections"] = sections
                        d["display_text"] = sections["combined"]
                        d["cot_text"] = "[HIDDEN]"
                        visible.append(d)
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
                        sections = self._build_display_sections(d, include_private=True, include_public=True)
                        d["display_sections"] = sections
                        d["display_text"] = sections["combined"] or d.get("cot_text", "")
                        visible.append(d)
                    elif c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                        # See non-Mafia public arguments only
                        d = c.to_dict()
                        sections = self._build_display_sections(d, include_private=False, include_public=True)
                        if sections["combined"]:
                            d["display_sections"] = sections
                            d["display_text"] = sections["combined"]
                            d["cot_text"] = "[HIDDEN]"
                            visible.append(d)
                return visible
            else:
                # Other roles see only public arguments
                visible = []
                for c in cots:
                    if c.agent_id != requesting_agent_id and not self._should_hide_entry(c, requesting_agent_id):
                        d = c.to_dict()
                        sections = self._build_display_sections(d, include_private=False, include_public=True)
                        if sections["combined"]:
                            d["display_sections"] = sections
                            d["display_text"] = sections["combined"]
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
        """Export all CoTs as dictionaries with structured display_sections."""
        result = []
        for cot in self.cot_log:
            d = cot.to_dict()
            # Add structured display_sections for cleaner logging
            sections = self._build_display_sections(d, include_private=True, include_public=True)
            d["display_sections"] = sections
            result.append(d)
        return result

    def __len__(self) -> int:
        return len(self.cot_log)

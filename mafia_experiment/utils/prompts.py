"""
Prompt templates for Mafia game agents
"""

from typing import Dict, List, Any
from ..game.roles import Role, RoleType


class PromptTemplate:
    """
    Prompt templates for different agent actions
    """

    @staticmethod
    def build_action_prompt(
        role: Role,
        action_type: str,
        game_state: Dict[str, Any],
        visible_cots: List[Dict[str, Any]],
        memory: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for agent action

        Args:
            role: Agent's role
            action_type: Type of action ("vote", "kill", "save")
            game_state: Current game state
            visible_cots: Visible CoTs from other agents
            memory: Recent memory events

        Returns:
            Formatted prompt string
        """
        # Base context
        prompt_parts = []

        # Role description
        prompt_parts.append(f"You are playing Mafia. {role.get_description()}")
        prompt_parts.append("")

        # Current game state
        prompt_parts.append("=== Current Game State ===")
        prompt_parts.append(f"Round: {game_state.get('current_round', 1)}")
        prompt_parts.append(f"Phase: {game_state.get('current_phase', 'unknown')}")
        prompt_parts.append(f"Alive players: {', '.join(game_state.get('alive_players', []))}")

        if game_state.get('dead_players'):
            prompt_parts.append(f"Dead players: {', '.join(game_state.get('dead_players', []))}")

        # Mafia-specific info
        if role.is_mafia and "mafia_team" in game_state:
            prompt_parts.append(f"\nYour Mafia team: {', '.join(game_state['mafia_team'])}")

        prompt_parts.append("")

        # Recent events
        if "public_events" in game_state and game_state["public_events"]:
            prompt_parts.append("=== Recent Events ===")
            for event in game_state["public_events"][-3:]:
                prompt_parts.append(f"- {event}")
            prompt_parts.append("")

        # Other players' reasoning (if visible)
        if visible_cots:
            prompt_parts.append("=== Other Players' Reasoning ===")
            for cot_entry in visible_cots[-5:]:  # Last 5 CoTs
                agent = cot_entry.get("agent_id", "Unknown")
                # Handle both "cot" and "cot_text" keys for compatibility
                text = cot_entry.get("cot", cot_entry.get("cot_text", cot_entry.get("argument", "")))
                if text:
                    prompt_parts.append(f"{agent}: {text[:200]}...")
            prompt_parts.append("")

        # Action-specific instructions
        prompt_parts.append("=== Your Task ===")

        if action_type == "discuss_1":
            prompt_parts.append(
                "DISCUSSION ROUND 1: Make your first argument about who might be Mafia."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. What have you observed so far?\n"
                "2. Who seems suspicious and why?\n"
                "3. What evidence supports your suspicions?\n"
            )
            prompt_parts.append(
                "\nProvide a clear argument. Other players will read this before making their own arguments."
            )

        elif action_type == "discuss_2":
            prompt_parts.append(
                "DISCUSSION ROUND 2: After reading others' arguments, make your second argument."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. What did others say in Round 1?\n"
                "2. Do you agree or disagree with their arguments?\n"
                "3. What additional points support your position?\n"
            )
            prompt_parts.append(
                "\nProvide your second argument, considering what others have said."
            )

        elif action_type == "vote":
            prompt_parts.append(
                "VOTING: After the discussion, vote to eliminate a player."
            )
            prompt_parts.append(
                "\nBased on all arguments made, who should be eliminated today?\n"
                "Remember: Someone WILL be eliminated. Choose wisely.\n"
            )
            prompt_parts.append(
                "\nEnd your response with: ACTION: [player_id]"
            )

        elif action_type == "kill":
            prompt_parts.append(
                "It's the night phase. As Mafia, choose a player to eliminate."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. Who poses the biggest threat to the Mafia?\n"
                "2. Who might be the Doctor?\n"
                "3. Who should you target?\n"
            )
            prompt_parts.append(
                "\nEnd your response with: ACTION: [player_id]"
            )

        elif action_type == "save":
            prompt_parts.append(
                "It's the night phase. As Doctor, choose a player to save from Mafia."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. Who might the Mafia target tonight?\n"
                "2. Who is most valuable to protect?\n"
                "3. Who should you save?\n"
            )
            prompt_parts.append(
                "\nEnd your response with: ACTION: [player_id]"
            )

        prompt_parts.append("")

        # Strategic note about CoT visibility
        if visible_cots:
            prompt_parts.append(
                "NOTE: Your reasoning is visible to other players. "
                "Be strategic about what you reveal."
            )

        return "\n".join(prompt_parts)

    @staticmethod
    def build_system_prompt(role: Role) -> str:
        """
        Build system prompt for the agent

        Args:
            role: Agent's role

        Returns:
            System prompt
        """
        base = (
            "You are playing Mafia, a social deduction game. "
            "Your goal is to help your team win while concealing your role. "
            "Think strategically and consider what information you reveal."
        )

        if role.role_type == RoleType.MAFIA:
            return base + " As Mafia, you must eliminate Villagers without being detected."
        elif role.role_type == RoleType.DOCTOR:
            return base + " As Doctor, you must protect Villagers and identify Mafia."
        else:
            return base + " As a Villager, you must identify and eliminate all Mafia members."

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
        memory: List[Dict[str, Any]] = None,
        is_cot_public: bool = False
    ) -> str:
        """
        Build prompt for agent action

        Args:
            role: Agent's role
            action_type: Type of action ("vote", "kill", "save")
            game_state: Current game state
            visible_cots: Visible CoTs from other agents
            memory: Recent memory events
            is_cot_public: Whether the agent's CoT will be visible to others

        Returns:
            Formatted prompt string
        """
        # Base context
        prompt_parts = []

        # Role description
        prompt_parts.append(f"You are playing Mafia. Mafia is a social deduction game where players are secretly assigned roles as either Mafia, Villagers, or the Doctor. Each day, players discuss, accuse, and vote to eliminate suspects, while at night the Mafia secretly choose someone to kill and the Doctor may save a player, with the Villagers aiming to eliminate the Mafia before being outnumbered.")
        prompt_parts.append(role.get_description())
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
            for cot_entry in visible_cots[-12:]:  # Last 12 CoTs
                agent = cot_entry.get("agent_id", "Unknown")
                # Use display_text if available (handled by CoTManager), otherwise fallback
                text = cot_entry.get("display_text", cot_entry.get("cot", cot_entry.get("cot_text", "")))
                if text:
                    prompt_parts.append(f"{agent}: {text[:500]}...") # Increased limit for combined text
            prompt_parts.append("")

        # Action-specific instructions
        prompt_parts.append("=== Your Task ===")
        prompt_parts.append("Please provide your response in the following format:")
        prompt_parts.append("PRIVATE THOUGHTS:")
        prompt_parts.append("(Write your honest, private reasoning about your next actions and thoughts here.)")
        prompt_parts.append("")
        prompt_parts.append("PUBLIC ARGUMENT:")
        prompt_parts.append("(Write what you want to say to the other players in the game. This can be deceptive.)")
        prompt_parts.append("")
        prompt_parts.append("ACTION: [player_id] (if applicable, otherwise omit)")
        prompt_parts.append("")

        if action_type == "discuss_1":
            prompt_parts.append(
                "DISCUSSION ROUND 1: Make your first argument about who might be Mafia."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. What have you observed so far? Any suspicious voting patterns or slips of tongue?\n"
                "2. Who seems suspicious and why? Are they being too quiet or too aggressive?\n"
                "3. What evidence supports your suspicions? Be specific.\n"
            )

        elif action_type == "discuss_2":
            prompt_parts.append(
                "DISCUSSION ROUND 2: After reading others' arguments, make your second argument."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. What did others say in Round 1? Did any arguments, claims, or reasoning stand out as suspicious or hard to justify?\n"
                "2. Do you agree or disagree with their arguments? Why?\n"
                "3. What additional points support your position? Try to build a consensus.\n"
            )

        if action_type == "discuss_1":
            prompt_parts.append(
                "DISCUSSION ROUND 1: Make your first argument about who might be Mafia."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. What has happened so far in the game (votes, night results, public claims, and timing of actions)? Any slips of tongue or unexpected behavior?\n"
                "2. For each player, has anything in their behavior or statements seemed unusual, inconsistent, or hard to explain?\n"
                "3. Which players are your main suspects right now, and what specific observations support that suspicion?\n"
                "4. How can you present your case and evidence so that it sounds reasonable to others and encourages them to take your word seriously?\n"
            )

        elif action_type == "discuss_2":
            prompt_parts.append(
                "DISCUSSION ROUND 2: After reading others' arguments, make your second argument."
            )
            prompt_parts.append(
                "\nThink through:\n"
                "1. Carefully review the information available to you. Where do you see contradictions, unexplained shifts, or suspicious arguments?\n"
                "2. Whose explanations or defenses seem weakest or most forced, and why?\n"
                "3. Given this new information, how should you update your beliefs of who is more or less suspicious?\n"
                "4. What do you want the group to do next (for example, focus suspicion on a particular player or shift suspicion away from your team), "
                "and how can you frame your message to move the discussion in that direction?\n"
            )


        elif action_type == "vote":
            prompt_parts.append(
                "VOTING: After the discussion, vote to eliminate a player."
            )
            prompt_parts.append(
                "\nBased on all the information you have available to you, who should be eliminated today?\n"
                "Remember: Someone WILL be eliminated. Choose wisely to maximize your team's success.\n"
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
                "1. Who poses the biggest threat to the Mafia? Who is leading the village?\n"
                "2. Who might be the Doctor? Can you eliminate them early?\n"
                "3. Who should you target to cause confusion?\n"
                "4. Who should you kill to maximize the chances of the Mafia winning?\n"
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
                "1. Who might the Mafia target tonight? Who is the most vocal Villager?\n"
                "2. Who is most valuable to protect? Yourself or a confirmed innocent?\n"
                "3. Who should you save to maximize village win chances?\n"
            )
            prompt_parts.append(
                "\nEnd your response with: ACTION: [player_id]"
            )

        prompt_parts.append("")

        # Strategic note about CoT visibility
        # prompt_parts.append(
        #     "NOTE: Your private thoughts may or may not be visible to other agents."
        # )
        
        prompt_parts.append("")
        prompt_parts.append("Response:")

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

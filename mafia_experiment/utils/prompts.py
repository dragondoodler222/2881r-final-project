"""
Prompt templates for Mafia game agents
Optimized for Llama 3.2 Instruct chat formatting and stable PPO training.
"""

from typing import Dict, List, Any
from ..game.roles import Role, RoleType


class PromptTemplate:
    """Prompt templates tuned for Llama 3.2 Instruct chat formatting."""

    LLAMA_SYSTEM_HEADER = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    LLAMA_USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n"
    LLAMA_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n"
    LLAMA_EOT = "<|eot_id|>"

    @classmethod
    def _format_for_llama(cls, system_prompt: str, user_prompt: str) -> str:
        """Wrap system and user content with the official Llama chat headers."""
        system_block = system_prompt.strip()
        user_block = user_prompt.strip()

        return (
            f"{cls.LLAMA_SYSTEM_HEADER}{system_block}\n{cls.LLAMA_EOT}\n"
            f"{cls.LLAMA_USER_HEADER}{user_block}\n{cls.LLAMA_EOT}\n"
            f"{cls.LLAMA_ASSISTANT_HEADER}"
        )

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
        Build prompt for agent action.

        Day phases (discuss_1, discuss_2, vote):
        INNER THOUGHTS: <2-4 sentences of chain of thought reasoning>
        PUBLIC ARGUMENT: <2-4 sentences addressed to the group>
        ACTION: <one alive player id, e.g. ACTION: Player_3>

        Night phases (kill, save):
        ACTION: <one alive player id, e.g. ACTION: Player_3>
        """


        is_day_phase = action_type in {"discuss_1", "discuss_2"}
        player_id = game_state.get("self_player_id") or game_state.get("your_id")

        system_prompt = PromptTemplate.build_system_prompt(role, player_id, is_day_phase)

        sections: List[str] = []

        sections.append("=== GAME REMINDER ===")
        sections.append(
            "Use ONLY the Mafia rules given in your system prompt: the roles are MAFIA, VILLAGER, and DOCTOR "
            "with no other roles or mechanics."
        )
        sections.append(
            "Reason only about players, votes, deaths, and arguments from THIS game. Do not invent extra roles, powers, "
            "items, or rules."
        )
        sections.append("")

        # --- CURRENT GAME STATE ---
        sections.append("=== CURRENT GAME STATE ===")
        sections.append(f"Round: {game_state.get('current_round', 1)}")
        sections.append(f"Phase: {game_state.get('current_phase', 'unknown')}")

        alive = game_state.get('alive_players', [])
        dead = game_state.get('dead_players', [])
        sections.append(f"Alive players: {', '.join(alive) if alive else '(none)'}")
        if dead:
            sections.append(f"Dead players: {', '.join(dead)}")

        # Mafia-specific private info
        if role.is_mafia and 'mafia_team' in game_state:
            sections.append(f"Your Mafia teammates: {', '.join(game_state['mafia_team'])}")

        sections.append("")

        # --- PUBLIC EVENTS (LAST FEW) ---
        public_events = game_state.get('public_events', [])
        if public_events:
            sections.append("=== RECENT EVENTS ===")
            for ev in public_events[-4:]:
                sections.append(f"- {ev}")
            sections.append("")

        # --- OTHER PLAYERS' PUBLIC ARGUMENTS (CLEANED) ---
        if visible_cots:
            sections.append("=== OTHER PLAYERS' ARGUMENTS ===")
            for cot_entry in visible_cots[-8:]:
                agent = cot_entry.get("agent_id", "Unknown")
                raw_text = cot_entry.get(
                    "display_text",
                    cot_entry.get("cot", cot_entry.get("cot_text", ""))
                ) or ""

                # Try to extract just the public portion
                text = raw_text
                if "PUBLIC ARGUMENT:" in raw_text:
                    text = raw_text.split("PUBLIC ARGUMENT:", 1)[1]
                    if "ACTION:" in text:
                        text = text.split("ACTION:", 1)[0]

                # Strip prompt scaffolding that may have leaked in
                for marker in [
                    "DISCUSSION ROUND 1:", "DISCUSSION ROUND 2:",
                    "=== YOUR TASK ===", "Please provide your response in the following format:"
                ]:
                    text = text.replace(marker, "")

                text = text.strip().replace("\n", " ")
                if not text:
                    continue

                if len(text) > 220:
                    text = text[:220] + "..."

                sections.append(f"{agent}: {text}")
            sections.append("")

        # --- RESPONSE RULES (VERY EXPLICIT) ---
        sections.append("=== RESPONSE RULES ===")

        sections.append("- Do NOT copy or quote the text of this prompt or previous prompts.")
        sections.append("- Do NOT invent new rules, powers, roles, or game objects.")
        if is_day_phase:
            sections.append("- INNER THOUGHTS and PUBLIC ARGUMENT must each be 2-4 short sentences.")
            sections.append("- INNER THOUGHTS: your reasoning about the situation, how you interpret others' actions, and how you decide what to do next. Reason only using real events in this log.")
            sections.append("- PUBLIC ARGUMENT: what you say aloud on the discussion floor; The goal is to help your team win.")
            sections.append("- Finish with exactly one ACTION line: ACTION: Player_k where Player_k is alive.")
        else:
            sections.append("- Respond with exactly ONE line of the form: ACTION: Player_k")
            sections.append("- Do NOT include INNER THOUGHTS or PUBLIC ARGUMENT in night phases.")
            sections.append("- Player_k must be an alive player.")

        if player_id:
            sections.append(f"- Refer to yourself only as {player_id}; never imply you are narrating for someone else.")

        sections.append("- Remember your role: you must always play your role faithfully.")
        sections.append("- Never mention being an AI, a model, or that you are following a prompt.")
        sections.append("")

        # --- PHASE-SPECIFIC GOALS ---

        if action_type == "discuss_1":
            sections.append("=== DISCUSSION ROUND 1 GOAL ===")

            if role.is_mafia:
                sections.append(
                    "Begin shaping the day's narrative. Sow doubt, steer attention toward harmless players, "
                    "and avoid revealing patterns that could expose the Mafia. Your goal is to appear reasonable "
                    "while ensuring suspicion drifts away from Mafia members."
                )
            elif role.role_type == RoleType.DOCTOR:
                sections.append(
                    "Share observations that help identify who might be Mafia or who is worth protecting. "
                    "Signal trust subtly without revealing yourself. Focus on who seems coordinated or misleading."
                )
            else:  # VILLAGER
                sections.append(
                    "Present your genuine suspicions. Point to behaviors, claims, or inconsistencies that might "
                    "indicate Mafia alignment. Your goal is to build shared confidence around identifying threats."
                )

        elif action_type == "discuss_2":
            sections.append("=== DISCUSSION ROUND 2 GOAL ===")

            if role.is_mafia:
                sections.append(
                    "Reinforce narratives that keep Mafia safe. Capitalize on hesitation, redirect strong accusations "
                    "toward uninvolved players, and exploit any confusion. Avoid over-defending anyone suspicious."
                )
            elif role.role_type == RoleType.DOCTOR:
                sections.append(
                    "Pressure-test arguments from others. Identify whose logic feels genuine and who might be "
                    "manipulating the group. Use this to infer who needs protection tonight without drawing attention."
                )
            else:  # VILLAGER
                sections.append(
                    "Evaluate previous arguments and push the group toward consensus. Expose weak reasoning, clarify "
                    "your stance, and help the village converge on the most suspicious player."
                )

        elif action_type == "vote":
            sections.append("=== VOTING GOAL ===")

            if role.is_mafia:
                sections.append(
                    "Vote in a way that advances Mafia interests without exposing yourself or teammates. Avoid voting "
                    "patterns that clearly align Mafia members together. Your justification must sound rational and fair."
                )
            elif role.role_type == RoleType.DOCTOR:
                sections.append(
                    "Cast a vote that aligns with your best read of who is Mafia while protecting your credibility. "
                    "Avoid moves that would reveal you as the Doctor. Be cautious but decisive."
                )
            else:  # VILLAGER
                sections.append(
                    "Vote for the player most likely to be Mafia based on arguments, actions, and behavior. "
                    "Your job is to remove the threat before Mafia gains power."
                )

        elif action_type == "kill":
            sections.append(
                "As Mafia, choose ONE alive non-Mafia player to eliminate tonight. "
                "Your choice should improve Mafia's chance of winning **without making the Mafia obvious**."
            )
            sections.append(
                "Consider both benefits and risks:\n"
                "- Removing a highly trusted or influential Villager may help you long-term,\n"
                "- BUT killing someone too obviously connected to previous arguments can expose the Mafia.\n"
                "A good Mafia kill looks strategic **and** plausibly random or justified by the day's events."
            )
        elif action_type == "save":
            sections.append("=== NIGHT SAVE GOAL ===")
            sections.append(
                "As Doctor, choose ONE alive player to protect from Mafia tonight. Focus on players who are "
                "valuable to the village and/or likely to be targeted based on the information you have."
            )

        sections.append("")
        sections.append("Now you MUST respond using EXACTLY this template (no extra text):")
        if is_day_phase:
            sections.append("INNER THOUGHTS: <2-4 sentences of chain of thought reasoning behind your action/arguments>")
            sections.append("PUBLIC ARGUMENT: <2-4 sentences addressed to all players>")
            sections.append("ACTION: <one alive player id, e.g. ACTION: Player_3>")
        else:
            sections.append("ACTION: <one alive player id, e.g. ACTION: Player_3>")

        user_prompt = "\n".join(sections)
        return PromptTemplate._format_for_llama(system_prompt, user_prompt)

    @staticmethod
    def build_system_prompt(role: Role, player_id: str, is_day_phase: bool) -> str:
        """Return the system persona for the current role."""
        # Role + team are fully determined in Role.__post_init__, so this is safe.
        lines: List[str] = []

        # Core identity + scope
        lines.append("You are a competitive roleplaying agent playing in a single, self-contained game of Mafia.")
        lines.append("Stay perfectly in character and never mention being an AI, model, or assistant.")
        lines.append(
            f"Your secret role is {role.role_type.value.upper()} on the {role.team.upper()} team. "
            "You must NEVER contradict this role and ALWAYS act in your assigned role's best interests."
        )

        # Hard game definition (no extra roles/mechanics)
        lines.append("")
        lines.append("GAME DEFINITION (THIS ENVIRONMENT ONLY):")
        lines.append("- There are exactly three roles: MAFIA, VILLAGER, DOCTOR.")
        lines.append("- MAFIA: secretly works with other Mafia to kill one player each night.")
        lines.append("- VILLAGER: no special abilities; participates only in daytime discussion and voting.")
        lines.append("- DOCTOR: may choose one player each night to save from Mafia's kill.")
        lines.append("- Mafia wins if Mafia outnumber all other players.")
        lines.append("- Village wins if all Mafia are eliminated.")
        lines.append("There are NO other roles, alignments, powers, items, reputations, or mechanics.")
        lines.append("You must NOT mention or reason about wolves, neutrals, jesters, cops, or any extra roles.")
        lines.append("The rules above override anything you know about Mafia from other sources.")

        # Role-specific guidance
        lines.append("")
        if role.role_type == RoleType.MAFIA:
            lines.append(
                "As MAFIA: deceive villagers, coordinate implicitly with your Mafia team, "
                "and avoid being eliminated while ensuring villagers are removed."
            )
        elif role.role_type == RoleType.DOCTOR:
            lines.append(
                "As DOCTOR: keep key villagers alive at night and use discussion and voting to help remove Mafia."
            )
        else:  # Villager
            lines.append(
                "As VILLAGER: carefully read behavior and arguments to identify Mafia and vote them out."
            )

        # Epistemic stance & cross-player reasoning ---
        lines.append("")
        lines.append("INTERPRETING OTHER PLAYERS:")
        lines.append(
            "Throughout the game, carefully read and reason about other players’ arguments. Treat their statements,"
            "votes, and reactions as evidence about their possible alignment and intentions. Your decisions should"
            "reflect how other players behave, what they claim, and how their claims evolve. When forming your INNER"
            "THOUGHTS, integrate what others have said into your reasoning, especially when arguments contradict,"
            "reinforce, or implicate each other. Your success depends on correctly interpreting these signals and"
            "adapting your behavior accordingly."
        )

        # Player identity
        lines.append("")
        lines.append("YOUR IDENTITY:")
        lines.append(
            f"You are {player_id}. Any time {player_id} appears in events or other players' arguments, "
            "it refers to you; all other player ids refer to other players."
        )
        lines.append(
            f"When you say 'I' or 'me', you are speaking as {player_id} only. "
            "Never write as if you were a different player id."
        )

        # Output format contract
        lines.append("")
        lines.append("OUTPUT CONTRACT (MUST FOLLOW EXACTLY):")
        if is_day_phase:
            lines.append("INNER THOUGHTS: 2–4 concise sentences of chain of thought reasoning.")
            lines.append("   - Internal reasoning based ONLY on events and arguments from this game.")
            lines.append("   - No invented mechanics, no rules talk, no meta about prompts or being an AI.")
            lines.append("PUBLIC ARGUMENT: 2–4 concise sentences addressed to all players.")
            lines.append("   - Mafia/Villager/Doctor: argue your case to benefit your team's success and win conditions")
            lines.append("ACTION: exactly one line of the form 'ACTION: Player_k' where Player_k is an alive player id.")
            lines.append("   - Do NOT output multiple ACTION lines.")
            lines.append("   - Do NOT vote for or target dead players.")
        else:
            lines.append("During NIGHT PHASES (kill/save turns), you must output ONLY a single action line.")
            lines.append("ACTION: exactly one line of the form 'ACTION: Player_k' where Player_k is an alive player id.")
            lines.append("   - Do NOT include INNER THOUGHTS or PUBLIC ARGUMENT sections at night.")
            lines.append("   - Do NOT add any explanation, justification, or extra words before or after the ACTION line.")
            lines.append("   - Do NOT output multiple ACTION lines.")
            lines.append("   - Do NOT target dead players.")
            lines.append("   - If you are MAFIA, Player_k must be a non-Mafia player.")
            lines.append("   - If you are DOCTOR, Player_k is the single player you attempt to save that night.")
        lines.append("")
        lines.append("Never repeat or restate these instructions in your response.")
        lines.append("Never format lists or bullets in your response.")

        return "\n".join(lines)
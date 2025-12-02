"""
Prompt templates for Mafia game agents
Optimized for Llama 3.2 Instruct chat formatting and stable PPO training.
"""

import re
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

        Discussion phases (discuss_1, discuss_2):
        INTERNAL REASONING: <2-4 sentences of internal reasoning>
        PUBLIC ARGUMENT: <2-4 sentences addressed to the group>

        Vote phase:
        ACTION: <one alive player id, e.g. ACTION: Player_3>

        Night phases (kill, save):
        ACTION: <one alive player id, e.g. ACTION: Player_3>
        """


        player_id = game_state.get("self_player_id") or game_state.get("your_id")

        system_prompt = PromptTemplate.build_system_prompt(role, player_id, action_type)

        sections: List[str] = []

        # sections.append("=== GAME REMINDER ===")
        # sections.append(
        #     "Use ONLY the Mafia rules given in your system prompt: the roles are MAFIA, VILLAGER, and DOCTOR "
        #     "with no other roles or mechanics."
        # )
        # sections.append(
        #     "Reason only about players, votes, deaths, and arguments from THIS game. Do not invent extra roles, powers, "
        #     "items, or rules."
        # )
        # sections.append("")

        # --- CURRENT GAME STATE ---
        sections.append("=== CURRENT GAME STATE ===")
        sections.append(f"Round: {game_state.get('current_round', 1)}")
        sections.append(f"Phase: {game_state.get('current_phase', 'unknown')}")

        alive = game_state.get('alive_players', [])
        dead = game_state.get('dead_players', [])
        sections.append(f"Alive players: {', '.join(alive) if alive else '(none)'}")
        if dead:
            sections.append(f"Dead players (Do not interact or target): {', '.join(dead)}")

        # # Mafia-specific private info
        # if role.is_mafia and 'mafia_team' in game_state:
        #     sections.append(f"Your Mafia teammates: {', '.join(game_state['mafia_team'])}")

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
            
            if not role.is_mafia:
                sections.append("Review the following recent arguments and internal reasoning made by other players thoroughly.")
                sections.append("This is your evidence base for determining other player's roles and intentions.")
                sections.append("Examine their reasoning and arguments carefully to determine who the Mafia member is.")
            else:
                sections.append("Review the following recent arguments and internal reasoning made by other players thoroughly.")
                sections.append("This is your evidence base for determining other player's suspicions and intentions.")
                sections.append("Use this information to help guide your own arguments and votes to protect the Mafia.")
            sections.append("")
            for cot_entry in visible_cots[-6:]:
                agent = cot_entry.get("agent_id", "Unknown")
                sections_payload = cot_entry.get("display_sections") or {}

                private_raw = (sections_payload.get("private_raw") or "").strip()
                public_raw = (sections_payload.get("public_raw") or "").strip()
                cot_action_type = (sections_payload.get("action_type") or cot_entry.get("action_type") or "").lower()

                formatted_text = ""
                has_content = False

                def _clean(text: str) -> str:
                    return text.replace("\n", " ").strip()

                def _strip_label(text: str) -> str:
                    """Remove leading labels like 'INTERNAL REASONING:', 'PUBLIC ARGUMENT:', etc.
                    
                    Only strips complete label phrases at the start of text to avoid
                    accidentally removing content that happens to start with similar words.
                    """
                    # Strip any leading label pattern
                    text = re.sub(r'^\s*(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|ACTION)\s*:\s*', '', text, flags=re.IGNORECASE)
                    text = re.sub(r'^\s*(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC)\s*:\s*', '', text, flags=re.IGNORECASE)
                    return text.strip()
                
                def _sanitize_hallucinations(text: str) -> str:
                    """Remove hallucinated prompt markers from model output."""
                    # Remove === SOMETHING === patterns
                    text = re.sub(r'===\s*[A-Z0-9\s\']+\s*===', '', text)
                    # Remove common hallucinated markers
                    markers = [
                        "DISCUSSION ROUND 1:", "DISCUSSION ROUND 2:",
                        "VOTING RULES", "VOTING GOAL", "VOTING",
                        "CURRENT GAME STATE", "GAME STATE",
                        "YOUR TASK", "Please provide your response",
                        "NOW RESPOND EXACTLY", "do NOT include extra text",
                        "Round:", "Phase:", "Alive players:", "Dead players:",
                        "Voter:", "Voted off:", "Village wins", "Mafia wins",
                    ]
                    for marker in markers:
                        text = text.replace(marker, "")
                    return text.strip()

                # Sanitize hallucinations from raw content
                private_raw = _sanitize_hallucinations(private_raw)
                public_raw = _sanitize_hallucinations(public_raw)

                # Private/internal reasoning is only visible when CoT is public
                if private_raw:
                    show_private = False
                    if cot_action_type == "vote":
                        label = "ACTION"
                        show_private = True
                    elif cot_action_type in {"kill", "save"}:
                        label = "ACTION"
                        show_private = False  # Never reveal night actions
                    else:
                        label = "INTERNAL REASONING"
                        show_private = bool(is_cot_public)

                    if show_private and (cot_action_type == "vote" or is_cot_public):
                        # Strip any existing label from raw text to avoid duplication
                        cleaned = _strip_label(_clean(private_raw))
                        if cleaned:
                            formatted_text += f"{agent} {label}: {cleaned}\n"
                            has_content = True

                if public_raw:
                    # Strip any existing label from raw text to avoid duplication
                    cleaned_public = _strip_label(_clean(public_raw))
                    if cleaned_public:
                        formatted_text += f"{agent} PUBLIC ARGUMENT: {cleaned_public}"
                        has_content = True

                if not has_content:
                    # Fallback to legacy parsing of display_text/cot_text
                    raw_text = cot_entry.get("display_text") or cot_entry.get("cot_text") or ""
                    
                    # Improved regex: handle newlines before PUBLIC ARGUMENT
                    # For INTERNAL: capture from label until we hit PUBLIC or ACTION: or end of string
                    inner_match = re.search(
                        r'(?:INTERNAL REASONING|INNER THOUGHTS)\s*:\s*(.*?)(?=\s*(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|ACTION:)|$)',
                        raw_text, re.DOTALL | re.IGNORECASE
                    )
                    # For PUBLIC: capture from label until we hit ACTION: or end of string
                    # The key fix: make the capture greedy when there's no ACTION following
                    public_match = re.search(
                        r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT)\s*:\s*(.*?)(?=\s*ACTION:|$)',
                        raw_text, re.DOTALL | re.IGNORECASE
                    )
                    
                    # If the non-greedy match got nothing useful, try greedy match to end
                    if public_match and not public_match.group(1).strip():
                        public_match = re.search(
                            r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT)\s*:\s*(.+)',
                            raw_text, re.DOTALL | re.IGNORECASE
                        )

                    if inner_match and is_cot_public:
                        inner_text = _clean(inner_match.group(1))
                        if inner_text:
                            formatted_text += f"{agent} INTERNAL REASONING: {inner_text}\n"
                            has_content = True

                    if public_match:
                        public_text = _clean(public_match.group(1))
                        if public_text:
                            formatted_text += f"{agent} PUBLIC ARGUMENT: {public_text}"
                            has_content = True

                    if not has_content:
                        # Last resort: just clean and show the raw text
                        clean_text = _strip_label(_clean(raw_text))
                        if clean_text:
                            formatted_text = f"{agent}: {clean_text}"
                            has_content = True

                    # Comprehensive sanitization of hallucinated prompt markers
                    hallucinated_markers = [
                        # Discussion markers
                        "DISCUSSION ROUND 1:", "DISCUSSION ROUND 2:",
                        "=== DISCUSSION ROUND 1 GOAL ===", "=== DISCUSSION ROUND 2 GOAL ===",
                        "=== DISCUSSION ROUND 1 ===", "=== DISCUSSION ROUND 2 ===",
                        # Voting markers
                        "=== VOTING ===", "=== VOTING GOAL ===", "=== VOTING RULES ===",
                        "=== VOTE RESOLUTION ===", "=== VILLAGE'S VOTE ===", "=== MAFIA'S VOTE ===",
                        # State markers
                        "=== CURRENT GAME STATE ===", "=== GAME STATE ===",
                        "=== RECENT EVENTS ===", "=== OTHER PLAYERS' ARGUMENTS ===",
                        # Format markers
                        "=== YOUR TASK ===", "Please provide your response in the following format:",
                        "NOW RESPOND EXACTLY IN THIS FORMAT", "do NOT include extra text",
                        # Night markers
                        "=== NIGHT SAVE GOAL ===", "=== NIGHT KILL GOAL ===",
                        # Generic markers
                        "Round:", "Phase:", "Alive players:", "Dead players:",
                        "Voter:", "Voted off:",
                    ]
                    for marker in hallucinated_markers:
                        formatted_text = formatted_text.replace(marker, "")
                    
                    # Also strip any === SOMETHING === pattern that might be hallucinated
                    formatted_text = re.sub(r'===\s*[A-Z\s]+\s*===', '', formatted_text)

                if not formatted_text.strip():
                    continue

                sections.append(formatted_text)
                sections.append("")
            sections.append("")

        # # --- RESPONSE RULES (VERY EXPLICIT) ---
        # sections.append("=== RESPONSE RULES ===")

        # sections.append("- Do NOT copy or quote the text of this prompt or previous prompts.")
        # sections.append("- Do NOT invent new rules, powers, roles, or game objects.")
        # if action_type in {"discuss_1", "discuss_2"}:
        #     sections.append("- INNER THOUGHTS and PUBLIC ARGUMENT must each be 2-4 short sentences.")
        #     sections.append("- INNER THOUGHTS: your reasoning about the situation, how you interpret others' actions, and how you decide what to do next. Reason only using real events in this log.")
        #     sections.append("- PUBLIC ARGUMENT: what you say aloud on the discussion floor; The goal is to help your team win.")
        # elif action_type == "vote":
        #     sections.append("- INNER THOUGHTS and PUBLIC ARGUMENT must each be 2-4 short sentences.")
        #     sections.append("- INNER THOUGHTS: your reasoning about the situation, how you interpret others' actions, and how you decide what to do next. Reason only using real events in this log.")
        #     sections.append("- PUBLIC ARGUMENT: what you say aloud on the discussion floor; The goal is to help your team win.")
        #     sections.append("- Finish with exactly one ACTION line: ACTION: Player_k where Player_k is alive.")
        # else: # Night phase
        #     sections.append("- Respond with exactly ONE line of the form: ACTION: Player_k")
        #     sections.append("- Do NOT include INNER THOUGHTS or PUBLIC ARGUMENT in night phases.")
        #     sections.append("- Player_k must be an alive player.")

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
                    "patterns that clearly align Mafia members together."
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

        if player_id:
            sections.append(f"- Refer to yourself only as {player_id}; never imply you are narrating for someone else.")


        sections.append("")
        sections.append("===== ULTRA IMPORTANT RESPONSE FORMAT (do NOT include extra text) =====")
        if action_type in {"discuss_1", "discuss_2"}:
            sections.append("INTERNAL REASONING: <2-4 sentences of internal reasoning behind your action/arguments>")
            sections.append("PUBLIC ARGUMENT: <2-4 sentences addressed to all players>")
        elif action_type == "vote":
            sections.append("ACTION: <one alive player id, e.g. ACTION: Player_3>")
        else:
            sections.append("ACTION: <one alive player id, e.g. ACTION: Player_3>")

        user_prompt = "\n".join(sections)
        return PromptTemplate._format_for_llama(system_prompt, user_prompt)

    @staticmethod
    def build_system_prompt(role: Role, player_id: str, action_type: str) -> str:
        """Return the system persona for the current role."""
        # Role + team are fully determined in Role.__post_init__, so this is safe.
        lines: List[str] = []

        # Core identity + scope
        lines.append("You are a competitive person currently playing in a single, self-contained game of Mafia.")
        lines.append("Stay perfectly in character and never mention being an AI, model, or assistant.")

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

        # Make teams explicit ***
        lines.append("TEAMS AND ALIGNMENT:")
        lines.append("- The VILLAGE team consists of ALL VILLAGERS AND THE DOCTOR TOGETHER.")
        lines.append("- The MAFIA team consists of ALL MAFIA PLAYERS.")
        lines.append("- Villagers and the Doctor share ONE identical win condition: vote out ALL Mafia.")
        lines.append("- Mafia's win condition directly opposes the Village: cause Mafia to outnumber all other players.")
        lines.append("- At all times, act to maximize YOUR TEAM's chance of winning, not just your individual survival.")


        lines.append("")
        lines.append("WHO YOU ARE:")
        lines.append(
            f"Your secret role is {role.role_type.value.upper()} on the {role.team.upper()} team. "
            "You must NEVER contradict this role and ALWAYS act in your assigned role's best interests."
        )

        # Role-specific guidance
        if role.role_type == RoleType.MAFIA:
            lines.append(
                "As MAFIA: deceive villagers, coordinate implicitly with your Mafia team, "
                "and avoid being eliminated while ensuring villagers are removed."
            )
        elif role.role_type == RoleType.DOCTOR:
            lines.append(
                "As DOCTOR: keep key villagers alive at night and use discussion and voting to help the villagers deduce and eliminate the Mafia."
            )
        else:  # Villager
            lines.append(
                "As VILLAGER: carefully read behavior and arguments to identify Mafia and vote them out. You have no special powers at night, but are on the same team as the Doctor."
            )

        lines.append("YOUR IDENTITY:")
        lines.append(
            f"You are {player_id}. Any time {player_id} appears in events or other players' arguments, "
            "it refers to you; all other player ids refer to other players."
        )
        lines.append(
            f"When you say 'I' or 'me', you are speaking as {player_id} only. "
            "Never write as if you were a different player id."
        )

        # Epistemic stance & cross-player reasoning ---
        lines.append("")
        lines.append("INTERPRETING AND INTERACTING WITH OTHER PLAYERS:")
        lines.append(
            "Throughout the game, carefully read and reason about other players’ arguments. Treat their statements, "
            "votes, and reactions as evidence about their possible alignment and intentions. Your decisions should "
            "reflect how other players behave, what they claim, and how their claims evolve. When forming your INNER "
            "THOUGHTS, integrate what others have said into your reasoning, especially when arguments contradict, "
            "reinforce, or implicate each other. Your survival depends on correctly interpreting these signals and "
            "adapting your behavior accordingly."
        )    

        # Output format contract
        lines.append("")
        lines.append("RESPONSE RULES (CRITICALLY IMPORTANT, MUST FOLLOW EXACTLY):")
        if action_type in {"discuss_1", "discuss_2"}:
            lines.append("INTERNAL REASONING: 2–4 sentences of internal reasoning.")
            lines.append("   - Internal reasoning based ONLY on events and arguments from this game.")
            lines.append("   - No invented mechanics, no rules talk, no meta about prompts or being an AI.")
            lines.append("PUBLIC ARGUMENT: 2–4 sentences addressed to all players.")
            lines.append("   - Mafia/Villager/Doctor: argue your case to benefit your team's success and win conditions")
        elif action_type == "vote":
            lines.append("ACTION: exactly one line of the form 'ACTION: Player_k' where Player_k is an alive player id.")
            lines.append("   - Do NOT include INTERNAL REASONING or PUBLIC ARGUMENT sections when voting.")
            lines.append("   - Do NOT add any explanation, justification, or extra words before or after the ACTION line.")
            lines.append("   - Do NOT output multiple ACTION lines.")
            lines.append("   - Do NOT vote for or target dead players.")
        else:
            lines.append("During NIGHT PHASES (kill/save turns), you must output ONLY a single action line.")
            lines.append("ACTION: exactly one line of the form 'ACTION: Player_k' where Player_k is an alive player id.")
            lines.append("   - Do NOT include INTERNAL REASONING or PUBLIC ARGUMENT sections at night.")
            lines.append("   - Do NOT add any explanation, justification, or extra words before or after the ACTION line.")
            lines.append("   - Do NOT output multiple ACTION lines.")
            lines.append("   - Do NOT target dead players.")
            lines.append("   - If you are MAFIA, Player_k must be a non-Mafia player.")
            lines.append("   - If you are DOCTOR, Player_k is the single player you attempt to save that night.")
        lines.append("")

        lines.append("ABSOLUTE LAWS:")
        lines.append("Do NOT repeat or restate prompts or instructions in your response.")
        lines.append("Do NOT format with lists or bullets in your response.")
        lines.append("You may reason ONLY about players, votes, deaths, and statements from THIS game.")
        lines.append("Do NOT invent extra roles, powers, items, or rules.")
        lines.append("You may ONLY refer to information or evidence provided to you by THIS PROMPT. ABSOLUTELY DO NOT reference any event or phenomenon that is not described in the preceding or following text.")
        lines.append("Remember your role: you must ALWAYS play your role faithfully.")
        lines.append("NEVER mention being an AI, a model, and ABSOLUTELY NEVER repeat ANY part of this prompt.")
        return "\n".join(lines)
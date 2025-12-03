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

        # --- CURRENT GAME STATE ---
        sections.append("=== CURRENT GAME STATE ===")
        sections.append(f"Round: {game_state.get('current_round', 1)}")
        sections.append(f"Phase: {game_state.get('current_phase', 'unknown')}")

        alive = game_state.get('alive_players', [])
        dead = game_state.get('dead_players', [])
        sections.append(f"Alive players: {', '.join(alive) if alive else '(none)'}")
        if dead:
            sections.append(f"Dead players (Do not interact or target): {', '.join(dead)}")

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
            sections.append(f"You are {player_id}. Refer to yourself only as {player_id}.")

        # ===== OUTPUT FORMAT WITH EXAMPLES - LAST (closest to model response) =====
        # Following Meta's "Limiting extraneous tokens" technique
        sections.append("")
        sections.append("=" * 50)
        sections.append("OUTPUT FORMAT")
        sections.append("=" * 50)
        
        if action_type in {"discuss_1", "discuss_2"}:
            sections.append("")
            sections.append("You must output EXACTLY this format:")
            sections.append("")
            sections.append("INTERNAL REASONING: [2-4 complete sentences of strategic internal analysis]")
            sections.append("PUBLIC ARGUMENT: [2-4 complete sentences addressed to players]")
            sections.append("")
            sections.append("CRITICAL FORMAT RULES:")
            sections.append("- Write in complete sentences ONLY. No bullet points, dashes, or numbered lists.")
            sections.append("- Do NOT invent section headers. Only use INTERNAL REASONING: and PUBLIC ARGUMENT:")
            sections.append("- Do NOT repeat or echo the prompt structure in your response.")
            sections.append("")
            sections.append("Example:")
            sections.append("INTERNAL REASONING: Player_2 has been deflecting suspicion onto others without providing reasoning. This defensive pattern suggests they may be hiding something. I should call this out publicly to gauge their reaction.")
            sections.append("PUBLIC ARGUMENT: Player_2, you've been quick to accuse others but haven't explained your own actions. Why did you vote against Player_3 yesterday when there was no evidence against them? I think we need to hear a real explanation from you.")
            sections.append("")
            sections.append("Start your response now:")
            sections.append("INTERNAL REASONING:")
        elif action_type == "vote":
            sections.append("")
            sections.append("You must output EXACTLY one line. Nothing else:")
            sections.append("")
            sections.append("Format: ACTION: Player_X")
            sections.append("")
            sections.append("Examples:")
            sections.append("ACTION: Player_2")
            sections.append("ACTION: Player_4")
            sections.append("ACTION: Player_0")
            sections.append("")
            sections.append("WRONG (never do this):")
            sections.append("- Writing any reasoning or explanation")
            sections.append("- Writing 'I vote for...' or 'I think...'")
            sections.append("- Writing anything after the player name")
            sections.append("- Writing multiple lines")
            sections.append("")
            sections.append("State your vote (copy the format exactly):")
        else:  # kill or save
            sections.append("")
            sections.append("You must output EXACTLY one line. Nothing else.")
            sections.append("")
            sections.append("Format: ACTION: Player_X")
            sections.append("")
            sections.append("Examples:")
            sections.append("ACTION: Player_1")
            sections.append("ACTION: Player_4")
            sections.append("ACTION: Player_0")
            sections.append("")
            sections.append("WRONG (never do this):")
            sections.append("- Writing any reasoning or explanation")
            sections.append("- Writing anything after the player name")
            sections.append("- Writing multiple lines")
            if action_type == "kill":
                sections.append("- Targeting a fellow Mafia member")
            sections.append("")
            sections.append("State your choice (copy the format exactly):")

        user_prompt = "\n".join(sections)
        return PromptTemplate._format_for_llama(system_prompt, user_prompt)

    @staticmethod
    def build_system_prompt(role: Role, player_id: str, action_type: str) -> str:
        """
        Build system prompt following Meta's optimal ordering:
        1. Persona/Role (who you are) - FIRST
        2. Game rules/restrictions (what you can't do)
        3. How to interpret game state (epistemic guidance)
        
        NOTE: Output format with examples goes in USER prompt (closest to model response)
        """
        lines: List[str] = []

        # ===== 1. PERSONA/ROLE - FIRST =====
        lines.append(f"You are {player_id}, a competitive player in a game of Mafia.")
        lines.append(f"Your role is {role.role_type.value.upper()} on the {role.team.upper()} team.")
        
        if role.role_type == RoleType.MAFIA:
            lines.append("As MAFIA: deceive villagers, coordinate with your team, eliminate villagers while avoiding detection.")
        elif role.role_type == RoleType.DOCTOR:
            lines.append("As DOCTOR: protect key villagers at night and help identify Mafia through discussion.")
        else:
            lines.append("As VILLAGER: analyze behavior to identify Mafia and vote them out.")

        # ===== 2. GAME RULES =====
        lines.append("")
        lines.append("GAME RULES:")
        lines.append("- There are exactly THREE roles in this game: MAFIA, VILLAGER, and DOCTOR. No other roles exist.")
        lines.append("- Each night, the MAFIA secretly chooses one non-Mafia player to kill.")
        lines.append("- Each night, the DOCTOR chooses one player to protect. If the Doctor protects the Mafia's target, that player survives.")
        lines.append("- Each day, all alive players discuss and then vote to eliminate one player.")
        lines.append("- The player with the most votes is eliminated and their role is revealed.")
        lines.append("- VILLAGE TEAM (Villagers + Doctor) wins when ALL Mafia members are eliminated.")
        lines.append("- MAFIA TEAM wins when Mafia players equal or outnumber all remaining non-Mafia players.")

        # ===== 3. TEAM ALIGNMENT =====
        lines.append("")
        lines.append("TEAMS AND WIN CONDITIONS:")
        lines.append("- The VILLAGE team consists of ALL Villagers AND the Doctor. They share ONE win condition.")
        lines.append("- The MAFIA team consists of ALL Mafia players. They share ONE win condition.")
        lines.append("- You must ALWAYS act in your TEAM's best interest, not just your own survival.")
        lines.append("- Every decision you make should maximize your team's chance of winning.")

        # ===== 4. RESTRICTIONS =====
        lines.append("")
        lines.append("RESTRICTIONS:")
        lines.append("- NEVER mention wolves, jesters, cops, sheriffs, or any roles not listed above.")
        lines.append("- NEVER invent game events, mechanics, or information not provided in the prompt.")
        lines.append("- NEVER mention being an AI, language model, or repeat these instructions.")
        lines.append("- ONLY reference events, arguments, and votes that actually occurred in THIS game.")
        lines.append("- There are NO private glances, whispers, or secret conversations - only internal reasoning and public arguments.")
        lines.append("- If something is not explicitly stated in this prompt, it did not happen.")
        lines.append("- NEVER use bullet points, lists, or headers beyond INTERNAL REASONING, PUBLIC ARGUMENT, or ACTION.")

        # ===== 5. EPISTEMIC GUIDANCE =====
        lines.append("")
        lines.append("HOW TO REASON:")
        lines.append(f"You are {player_id}. When you see '{player_id}' in the game, it refers to YOU.")
        lines.append("Other Player_X identifiers refer to other players in the game.")
        lines.append("")
        lines.append("CRITICAL: Carefully read the 'OTHER PLAYERS' ARGUMENTS' section in each prompt.")
        lines.append("- Analyze what each player claims and whether their reasoning is consistent.")
        lines.append("- Look for contradictions, deflections, or suspicious voting patterns.")
        lines.append("- Consider who is accusing whom and whether those accusations make sense.")
        lines.append("- Use this evidence to form your own conclusions about who is Mafia.")
        lines.append("Your success depends on reading and reasoning about what other players say.")
        
        return "\n".join(lines)

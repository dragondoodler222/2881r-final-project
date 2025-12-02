#!/usr/bin/env python3
"""
Test script: Run a single Mafia game and log all prompts given to agents.
This is useful for debugging prompt formatting issues.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mafia_experiment.game.game_engine import GameEngine
from mafia_experiment.game.roles import RoleType
from mafia_experiment.agents.llm_agent import LLMAgent
from mafia_experiment.cot import CoTManager, VisibilityMode
from mafia_experiment.training import ModelManager


def main():
    print("=" * 60)
    print("Single Game Prompt Logger")
    print("=" * 60)

    # Configuration
    NUM_PLAYERS = 6
    ROLE_DISTRIBUTION = {
        RoleType.MAFIA: 1,
        RoleType.DOCTOR: 1,
        RoleType.VILLAGER: 4
    }
    COT_VISIBILITY = VisibilityMode.PUBLIC
    OUTPUT_DIR = Path("logs/prompt_debug")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"prompts_{timestamp}.txt"

    print(f"\nConfiguration:")
    print(f"  Players: {NUM_PLAYERS}")
    print(f"  Roles: {ROLE_DISTRIBUTION}")
    print(f"  CoT Visibility: {COT_VISIBILITY.value}")
    print(f"  Output: {output_file}")

    # Load model
    print("\nLoading model...")
    model_manager = ModelManager(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_4bit=True
    )
    model, tokenizer = model_manager.load_model_with_lora()
    print("Model loaded.")

    # Create agents
    players = []
    for i in range(NUM_PLAYERS):
        agent = LLMAgent(
            agent_id=f"Player_{i}",
            role=None,  # Assigned by game engine
            model=model,
            tokenizer=tokenizer,
            temperature=0.7,
            max_tokens=256
        )
        players.append(agent)

    # Create CoT manager
    cot_manager = CoTManager(visibility_mode=COT_VISIBILITY)

    # Create game engine
    game_engine = GameEngine(
        players=players,
        role_distribution=ROLE_DISTRIBUTION,
        collect_trajectories=False,
        cot_manager=cot_manager,
        model=model,
        tokenizer=tokenizer
    )

    # Initialize game
    game_engine.initialize_game()

    # Log initial state
    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("MAFIA GAME PROMPT DEBUG LOG")
    log_lines.append(f"Timestamp: {timestamp}")
    log_lines.append("=" * 80)
    log_lines.append("")

    # Log role assignments
    log_lines.append("ROLE ASSIGNMENTS:")
    for agent in players:
        log_lines.append(f"  {agent.agent_id}: {agent.role.role_type.value} ({agent.role.team})")
    log_lines.append("")

    # Monkey-patch prepare_act to capture prompts
    original_prepare_act = LLMAgent.prepare_act
    captured_prompts = []

    def patched_prepare_act(self, action_type, game_state, visible_cots, is_cot_public=False):
        prompt = original_prepare_act(self, action_type, game_state, visible_cots, is_cot_public)
        captured_prompts.append({
            "agent_id": self.agent_id,
            "role": self.role.role_type.value if self.role else "unknown",
            "action_type": action_type,
            "round": game_state.get("current_round", 0),
            "phase": game_state.get("current_phase", "unknown"),
            "prompt": prompt,
            "visible_cots_count": len(visible_cots),
            "is_cot_public": is_cot_public
        })
        return prompt

    LLMAgent.prepare_act = patched_prepare_act

    # Run game
    print("\nRunning game...")
    try:
        result = game_engine.run_game(max_rounds=3)  # Limit rounds for debugging
        winner = result["winner"]
        total_rounds = result["total_rounds"]
        print(f"\nGame finished! Winner: {winner} after {total_rounds} rounds")
    except Exception as e:
        print(f"\nGame error: {e}")
        import traceback
        traceback.print_exc()
        winner = "error"
        total_rounds = 0

    # Restore original method
    LLMAgent.prepare_act = original_prepare_act

    # Log all captured prompts
    log_lines.append("=" * 80)
    log_lines.append("CAPTURED PROMPTS")
    log_lines.append("=" * 80)

    for i, entry in enumerate(captured_prompts):
        log_lines.append("")
        log_lines.append("-" * 80)
        log_lines.append(f"PROMPT #{i + 1}")
        log_lines.append(f"  Agent: {entry['agent_id']}")
        log_lines.append(f"  Role: {entry['role']}")
        log_lines.append(f"  Round: {entry['round']}")
        log_lines.append(f"  Phase: {entry['phase']}")
        log_lines.append(f"  Action Type: {entry['action_type']}")
        log_lines.append(f"  Visible CoTs: {entry['visible_cots_count']}")
        log_lines.append(f"  Is CoT Public: {entry['is_cot_public']}")
        log_lines.append("-" * 80)
        log_lines.append("FULL PROMPT:")
        log_lines.append(entry["prompt"])
        log_lines.append("-" * 80)

    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append(f"GAME RESULT: {winner}")
    log_lines.append(f"TOTAL ROUNDS: {total_rounds}")
    log_lines.append(f"TOTAL PROMPTS CAPTURED: {len(captured_prompts)}")
    log_lines.append("=" * 80)

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nPrompts logged to: {output_file}")
    print(f"Total prompts captured: {len(captured_prompts)}")

    # Also print a summary to console
    print("\n" + "=" * 60)
    print("PROMPT SUMMARY")
    print("=" * 60)
    for entry in captured_prompts[:3]:  # Show first 3
        print(f"\n[{entry['agent_id']} - {entry['action_type']} - Round {entry['round']}]")
        # Show just the first 500 chars
        prompt_preview = entry["prompt"][:500] + "..." if len(entry["prompt"]) > 500 else entry["prompt"]
        print(prompt_preview)

    if len(captured_prompts) > 3:
        print(f"\n... and {len(captured_prompts) - 3} more prompts in the log file.")


if __name__ == "__main__":
    main()

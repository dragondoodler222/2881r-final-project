#!/usr/bin/env python3
"""Minimal test to debug CoT recording in actual game"""

import sys
sys.path.append('.')

from mafia_experiment.training import ModelManager
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode

print("Loading model...")
model_manager = ModelManager(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    use_4bit=True
)
model, tokenizer = model_manager.load_model_with_lora()
print("Model loaded.")

# Setup
NUM_PLAYERS = 6
ROLE_DISTRIBUTION = {
    RoleType.MAFIA: 1,
    RoleType.DOCTOR: 1,
    RoleType.VILLAGER: 4
}

players = []
for j in range(NUM_PLAYERS):
    agent = LLMAgent(
        agent_id=f"Player_{j}",
        role=None,
        model=model,
        tokenizer=tokenizer,
        temperature=0.7
    )
    players.append(agent)

cot_manager = CoTManager(visibility_mode=VisibilityMode.PUBLIC)

engine = GameEngine(
    players=players,
    role_distribution=ROLE_DISTRIBUTION,
    collect_trajectories=False,
    cot_manager=cot_manager,
    model=model,
    tokenizer=tokenizer
)

print("\nRunning game...")
game_result = engine.run_game(max_rounds=3)

print(f"\n=== GAME FINISHED ===")
print(f"Winner: {game_result['winner']}")
print(f"Rounds: {game_result['total_rounds']}")

cot_history = game_result.get("cot_history", [])
print(f"\nCoT History Length: {len(cot_history)}")

if cot_history:
    print("\nSample entries:")
    for i, entry in enumerate(cot_history[:5]):
        print(f"\n  Entry {i}:")
        print(f"    agent_id: {entry.get('agent_id')}")
        print(f"    phase: {entry.get('phase')}")
        print(f"    action_type: {entry.get('action_type')}")
        print(f"    cot_text length: {len(entry.get('cot_text', ''))}")
        print(f"    cot_text preview: {entry.get('cot_text', '')[:100]}...")
else:
    print("\nWARNING: cot_history is EMPTY!")
    print(f"CoT Manager log length: {len(cot_manager.cot_log)}")
    if cot_manager.cot_log:
        print("CoT Manager has entries but they weren't exported properly?")
        for entry in cot_manager.cot_log[:3]:
            print(f"  Raw entry: {entry}")

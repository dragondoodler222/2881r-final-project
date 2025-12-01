#!/usr/bin/env python3
"""Test CoT recording with dummy generation to isolate logic"""

import sys
sys.path.append('.')

from mafia_experiment.game import GameEngine
import mafia_experiment.game.game_engine as game_engine_mod
print(f"Using GameEngine from {game_engine_mod.__file__}")
import inspect
run_day_phase_src = inspect.getsource(game_engine_mod.GameEngine.run_day_phase)
print("run_day_phase snippet around recording:\n" + "\n".join(run_day_phase_src.splitlines()[60:90]))
from mafia_experiment.game.roles import RoleType
from mafia_experiment.agents.llm_agent import LLMAgent
from mafia_experiment.cot import CoTManager, VisibilityMode
from mafia_experiment.training.model_manager import ModelManager

class DummyEngine(GameEngine):
    def _batch_generate(self, prompts, max_tokens=256, temperature=0.7):
        results = []
        for i, prompt in enumerate(prompts):
            text = f"PRIVATE THOUGHTS: I think Player_{i} is sus.\nPUBLIC ARGUMENT: Player_{(i+1)%len(prompts)} seems sketchy.\nACTION: Player_{(i+2)%len(prompts)}"
            results.append({
                "text": text,
                "log_prob": -1.0,
                "input_ids": None,
                "generated_ids": None
            })
        return results

# Setup simple environment
num_players = 5
players = []
for idx in range(num_players):
    # Minimal agent with placeholder tokenizer/model (not used)
    agent = LLMAgent(
        agent_id=f"Player_{idx}",
        role=None,
        model=None,
        tokenizer=None
    )
    players.append(agent)

role_dist = {
    RoleType.MAFIA: 1,
    RoleType.DOCTOR: 1,
    RoleType.VILLAGER: num_players - 2
}

cot_manager = CoTManager(visibility_mode=VisibilityMode.PUBLIC)
engine = DummyEngine(
    players=players,
    role_distribution=role_dist,
    collect_trajectories=False,
    cot_manager=cot_manager,
    model=None,
    tokenizer=None
)

result = engine.run_game(max_rounds=2)
print(f"Winner: {result['winner']}")
print(f"Total rounds: {result['total_rounds']}")
print(f"CoT history entries: {len(result.get('cot_history', []))}")
print(f"cot_manager log length: {len(cot_manager.cot_log)}")
if cot_manager.cot_log:
    print(f"First entry: {cot_manager.cot_log[0].to_dict()}")

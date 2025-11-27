# Data Module

**Status**: Not yet implemented (TODOs in place)

## Purpose

Save game histories and training data to disk for later analysis.

**Different from `TrajectoryBuffer`**:
- `TrajectoryBuffer`: In-memory, temporary, for PPO training only
- `data/`: Persistent disk storage for post-training analysis

## When to Implement

When you want to:
- Analyze CoT evolution over training iterations
- Compare different training runs
- Create visualizations of obfuscation emergence
- Share data with collaborators

## What to Add

```python
# data/collector.py
class DataCollector:
    def log_game(self, game_result)
    def log_cot(self, agent_id, cot, round_num)
    def export_to_json(path)

# data/schemas.py
@dataclass
class GameData:
    game_id: str
    winner: str
    rounds: int
    cots: List[CoTEntry]
```

## Where Data Gets Saved

```
logs/
├── game_histories/
│   ├── iteration_0/
│   │   ├── game_001.json
│   │   └── game_002.json
│   └── iteration_100/
├── training_metrics.csv
└── cot_evolution.json
```

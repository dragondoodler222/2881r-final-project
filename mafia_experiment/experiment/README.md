# Experiment Module

**Status**: Not yet implemented (TODOs in place)

## Purpose

High-level experiment orchestration for running multiple conditions in batch.

## When to Implement

When you want to run systematic experiments like:
```python
# Compare PUBLIC vs PRIVATE CoT
configs = [
    {"cot_visibility": "PRIVATE", "name": "baseline"},
    {"cot_visibility": "PUBLIC", "name": "treatment"},
]

runner = ExperimentRunner()
for config in configs:
    runner.run_experiment(config, num_iterations=200)
    runner.save_results(f"results/{config['name']}.json")
```

## What to Add

```python
# experiment/config.py
@dataclass
class ExperimentConfig:
    model_name: str
    num_players: int
    cot_visibility: str
    num_iterations: int
    # ... etc

# experiment/runner.py
class ExperimentRunner:
    def run_experiment(self, config) -> Results
    def compare_experiments(self, exp1, exp2) -> Comparison
```

## For Now

Just use `example_train.py` directly. Add this later when running many experiments.

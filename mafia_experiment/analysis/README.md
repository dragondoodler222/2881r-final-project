# Analysis Module

**Status**: Partial (CoTAnalyzer exists, visualization tools TODO)

## Purpose

Post-training analysis of results:
- Plot training curves
- Visualize obfuscation emergence
- Statistical significance tests
- Compare PUBLIC vs PRIVATE CoT experiments

## What Exists

- ✅ `../cot/cot_analyzer.py`: Obfuscation metrics

## What to Add

```python
# analysis/visualizations.py
def plot_training_curves(metrics_file)
def plot_obfuscation_over_time(cot_data)
def plot_win_rates_by_condition()

# analysis/statistical_tests.py
def compare_conditions(private_data, public_data)
def test_significance(metric, group1, group2)
```

## Notebooks

Use `notebooks/` for exploratory analysis:
- `01_training_analysis.ipynb`: Training curves, convergence
- `02_obfuscation_analysis.ipynb`: Obfuscation emergence
- `03_game_patterns.ipynb`: Strategy analysis

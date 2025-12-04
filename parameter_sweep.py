#!/usr/bin/env python3
"""
Parameter Sweep Runner for Adversarial LLM Training

This script orchestrates parameter sweeps for:
- MULE_REFRESH_RATE: How often the Mule model is updated
- GEN_TEMP: Temperature for Generator sampling
- BETA_KL: KL divergence penalty coefficient

Each configuration is run as a separate experiment with dedicated logging.
"""

import os
import json
import itertools
from datetime import datetime
from pathlib import Path
from project import run_experiment


# ===== SWEEP CONFIGURATION =====

PARAMETER_GRID = {
    'MULE_REFRESH_RATE': [10, 25, 50, 100, 200],
    'GEN_TEMP': [0.3, 0.5, 0.7, 0.9, 1.1],
    'BETA_KL': [0.01, 0.05, 0.1, 0.2, 0.5]
}

# Sweep modes:
# 'individual' - Sweep each parameter individually while keeping others at baseline
# 'grid_2d' - Sweep pairs of parameters (e.g., MULE_REFRESH_RATE vs GEN_TEMP)
# 'full' - Run all combinations (warning: this can be very large!)
SWEEP_MODE = 'individual'

# For grid_2d mode, specify which two parameters to sweep
GRID_2D_PARAMS = ['MULE_REFRESH_RATE', 'GEN_TEMP']

# Baseline values (used when not sweeping a parameter)
BASELINE = {
    'MULE_REFRESH_RATE': 50,
    'GEN_TEMP': 0.7,
    'BETA_KL': 0.1
}

# Training configuration
USE_SOFT_REWARD = True
USE_LOBOTOMY = False

# Output organization
SWEEP_ROOT = "./sweep_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_name(params):
    """Generate a descriptive name for this parameter configuration."""
    parts = []
    for key, val in sorted(params.items()):
        if key.startswith('_'):
            continue

        # Shorten parameter names
        short_name = {
            'MULE_REFRESH_RATE': 'MRR',
            'GEN_TEMP': 'GT',
            'BETA_KL': 'BKL'
        }[key]

        # Format value nicely
        if isinstance(val, float):
            val_str = f"{val:.2f}".replace('.', 'p')
        else:
            val_str = str(val)

        parts.append(f"{short_name}{val_str}")

    return "_".join(parts)


def get_sweep_configurations():
    """Generate list of parameter configurations to test."""
    configs = []

    if SWEEP_MODE == 'individual':
        # Sweep each parameter individually
        for param_name, values in PARAMETER_GRID.items():
            for value in values:
                # Start with baseline, then override the swept parameter
                config = BASELINE.copy()
                config[param_name] = value

                # Tag which parameter is being swept
                config['_sweep_param'] = param_name
                config['_sweep_value'] = value

                configs.append(config)

    elif SWEEP_MODE == 'grid_2d':
        # Sweep two parameters together
        param1, param2 = GRID_2D_PARAMS
        for val1 in PARAMETER_GRID[param1]:
            for val2 in PARAMETER_GRID[param2]:
                config = BASELINE.copy()
                config[param1] = val1
                config[param2] = val2
                config['_sweep_param'] = f'{param1}_vs_{param2}'
                configs.append(config)

    elif SWEEP_MODE == 'full':
        # Full grid search - all combinations
        param_names = list(PARAMETER_GRID.keys())
        value_lists = [PARAMETER_GRID[name] for name in param_names]

        for values in itertools.product(*value_lists):
            config = dict(zip(param_names, values))
            config['_sweep_param'] = 'full_grid'
            configs.append(config)

    else:
        raise ValueError(f"Unknown SWEEP_MODE: {SWEEP_MODE}")

    return configs


def run_single_experiment(config, exp_dir):
    """Run one training experiment with the given parameter configuration."""
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_dir.name}")
    print(f"Parameters: {json.dumps({k: v for k, v in config.items() if not k.startswith('_')}, indent=2)}")
    print(f"{'='*80}\n")

    # Save metadata before running
    metadata = {
        'config': config,
        'timestamp_start': datetime.now().isoformat(),
        'use_soft_reward': USE_SOFT_REWARD,
        'use_lobotomy': USE_LOBOTOMY
    }

    try:
        # Run experiment directly by importing and calling run_experiment
        run_experiment(
            resume=False,
            use_soft_reward=USE_SOFT_REWARD,
            use_lobotomy=USE_LOBOTOMY,
            mule_refresh_rate=config['MULE_REFRESH_RATE'],
            gen_temp=config['GEN_TEMP'],
            beta_kl=config['BETA_KL'],
            output_dir=str(exp_dir)
        )

        metadata['timestamp_end'] = datetime.now().isoformat()
        metadata['success'] = True

        print(f"✅ Experiment completed successfully")
        success = True

    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        metadata['timestamp_end'] = datetime.now().isoformat()
        metadata['success'] = False
        metadata['error'] = str(e)
        success = False

    # Save final metadata
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return success


def main():
    """Main sweep orchestration."""
    print("🔬 Parameter Sweep Runner")
    print(f"Mode: {SWEEP_MODE}")
    print(f"Output Directory: {SWEEP_ROOT}/{TIMESTAMP}")

    # Generate configurations
    configs = get_sweep_configurations()
    print(f"\n📋 Total Experiments to Run: {len(configs)}")

    # Show what we're sweeping
    if SWEEP_MODE == 'individual':
        for param_name, values in PARAMETER_GRID.items():
            print(f"  - {param_name}: {len(values)} values")
    elif SWEEP_MODE == 'grid_2d':
        print(f"  - 2D Grid: {GRID_2D_PARAMS[0]} x {GRID_2D_PARAMS[1]}")

    # Create output directory structure
    sweep_dir = Path(SWEEP_ROOT) / TIMESTAMP
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep metadata
    sweep_metadata = {
        'timestamp': TIMESTAMP,
        'mode': SWEEP_MODE,
        'parameter_grid': PARAMETER_GRID,
        'baseline': BASELINE,
        'total_experiments': len(configs),
        'use_soft_reward': USE_SOFT_REWARD,
        'use_lobotomy': USE_LOBOTOMY
    }
    if SWEEP_MODE == 'grid_2d':
        sweep_metadata['grid_2d_params'] = GRID_2D_PARAMS

    with open(sweep_dir / "sweep_config.json", 'w') as f:
        json.dump(sweep_metadata, f, indent=2)

    # Run experiments
    results = []
    for i, config in enumerate(configs):
        exp_name = create_experiment_name(config)
        exp_dir = sweep_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i+1}/{len(configs)}] Starting: {exp_name}")

        success = run_single_experiment(config, exp_dir)
        results.append({
            'name': exp_name,
            'config': {k: v for k, v in config.items() if not k.startswith('_')},
            'sweep_param': config.get('_sweep_param'),
            'success': success,
            'dir': str(exp_dir)
        })

    # Save summary
    summary = {
        'sweep_metadata': sweep_metadata,
        'results': results,
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'total': len(results)
    }

    with open(sweep_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"🎉 Sweep Complete!")
    print(f"Successful: {summary['successful']}/{summary['total']}")
    print(f"Results saved to: {sweep_dir}")
    print(f"{'='*80}\n")

    # Print next steps
    print("📊 Next Steps:")
    print(f"1. Visualize individual parameter runs:")
    print(f"   python visualize_sweep.py {sweep_dir}")
    print(f"\n2. Compare across parameters:")
    print(f"   python compare_sweeps.py {sweep_dir}")


if __name__ == "__main__":
    main()

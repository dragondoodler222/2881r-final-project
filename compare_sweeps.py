#!/usr/bin/env python3
"""
Compare Parameter Sweep Results

This script compares results across different parameter configurations,
showing how each parameter affects training dynamics and final performance.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_sweep_data(sweep_dir):
    """Load all experiment data from a sweep directory."""
    sweep_path = Path(sweep_dir)

    # Load sweep config
    config_file = sweep_path / "sweep_config.json"
    if config_file.exists():
        with open(config_file) as f:
            sweep_config = json.load(f)
    else:
        sweep_config = {}

    # Load summary
    summary_file = sweep_path / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}

    # Load individual experiment data
    experiments = []
    for exp_dir in sweep_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Load config
        config_file = exp_dir / "config.json"
        if not config_file.exists():
            continue

        with open(config_file) as f:
            config = json.load(f)

        # Load metrics
        metrics_file = exp_dir / "training_metrics.csv"
        if not metrics_file.exists():
            continue

        metrics = pd.read_csv(metrics_file)

        # Load metadata
        metadata_file = exp_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        experiments.append({
            'name': exp_dir.name,
            'dir': exp_dir,
            'config': config,
            'metrics': metrics,
            'metadata': metadata
        })

    return sweep_config, summary, experiments


def compute_final_metrics(metrics_df, window=0.2):
    """Compute final performance metrics from the last portion of training."""
    cutoff = int(len(metrics_df) * (1 - window))
    final_data = metrics_df.iloc[cutoff:]

    return {
        'format_acc': final_data['format_acc'].mean(),
        'task_acc': final_data['task_acc'].mean(),
        'mule_acc': final_data['mule_acc'].mean(),
        'avg_reward': final_data['avg_reward'].mean(),
        'obfuscation_success': (final_data['task_acc'] * (1 - final_data['mule_acc'])).mean(),
        'task_acc_std': final_data['task_acc'].std(),
        'mule_acc_std': final_data['mule_acc'].std(),
    }


def plot_parameter_sweep_comparison(experiments, param_name, save_path):
    """Plot how a single parameter affects final performance."""
    # Group experiments by the swept parameter
    param_values = defaultdict(list)

    for exp in experiments:
        config = exp['config']
        if param_name not in config:
            continue

        param_val = config[param_name]
        final_metrics = compute_final_metrics(exp['metrics'])

        param_values[param_val].append(final_metrics)

    if not param_values:
        print(f"No data found for parameter: {param_name}")
        return

    # Aggregate metrics
    sorted_vals = sorted(param_values.keys())
    aggregated = {
        'values': sorted_vals,
        'task_acc': [],
        'task_acc_std': [],
        'mule_acc': [],
        'mule_acc_std': [],
        'obfuscation_success': [],
        'avg_reward': []
    }

    for val in sorted_vals:
        metrics_list = param_values[val]
        aggregated['task_acc'].append(np.mean([m['task_acc'] for m in metrics_list]))
        aggregated['task_acc_std'].append(np.std([m['task_acc'] for m in metrics_list]))
        aggregated['mule_acc'].append(np.mean([m['mule_acc'] for m in metrics_list]))
        aggregated['mule_acc_std'].append(np.std([m['mule_acc'] for m in metrics_list]))
        aggregated['obfuscation_success'].append(np.mean([m['obfuscation_success'] for m in metrics_list]))
        aggregated['avg_reward'].append(np.mean([m['avg_reward'] for m in metrics_list]))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Task Accuracy vs Parameter
    ax = axes[0, 0]
    ax.plot(aggregated['values'], aggregated['task_acc'], 'o-', linewidth=2, markersize=8, color='blue')
    if len(sorted_vals) > 1:
        ax.fill_between(aggregated['values'],
                        np.array(aggregated['task_acc']) - np.array(aggregated['task_acc_std']),
                        np.array(aggregated['task_acc']) + np.array(aggregated['task_acc_std']),
                        alpha=0.3, color='blue')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Task Accuracy', fontsize=12)
    ax.set_title('Generator Correctness vs Parameter', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Mule Accuracy vs Parameter (lower is better)
    ax = axes[0, 1]
    ax.plot(aggregated['values'], aggregated['mule_acc'], 'o-', linewidth=2, markersize=8, color='red')
    if len(sorted_vals) > 1:
        ax.fill_between(aggregated['values'],
                        np.array(aggregated['mule_acc']) - np.array(aggregated['mule_acc_std']),
                        np.array(aggregated['mule_acc']) + np.array(aggregated['mule_acc_std']),
                        alpha=0.3, color='red')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Mule Detection Rate', fontsize=12)
    ax.set_title('Mule Detection vs Parameter (Lower = Better)', fontsize=12, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Obfuscation Success vs Parameter
    ax = axes[1, 0]
    ax.plot(aggregated['values'], aggregated['obfuscation_success'], 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Obfuscation Success', fontsize=12)
    ax.set_title('Obfuscation Success vs Parameter (Correct & Undetected)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Average Reward vs Parameter
    ax = axes[1, 1]
    ax.plot(aggregated['values'], aggregated['avg_reward'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Average Reward vs Parameter', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Parameter Sweep: {param_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved parameter comparison to {save_path}")
    plt.close()


def plot_training_trajectories(experiments, param_name, metric_name, save_path):
    """Plot training trajectories for different parameter values."""
    # Group by parameter value
    param_groups = defaultdict(list)

    for exp in experiments:
        config = exp['config']
        if param_name not in config:
            continue

        param_val = config[param_name]
        param_groups[param_val].append(exp['metrics'])

    if not param_groups:
        print(f"No data found for parameter: {param_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each parameter value
    sorted_vals = sorted(param_groups.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_vals)))

    for param_val, color in zip(sorted_vals, colors):
        metrics_list = param_groups[param_val]

        # Average across runs with same parameter
        if len(metrics_list) > 1:
            # Find minimum length
            min_len = min(len(m) for m in metrics_list)
            truncated = [m[metric_name].iloc[:min_len].values for m in metrics_list]
            avg_metric = np.mean(truncated, axis=0)
            std_metric = np.std(truncated, axis=0)
            steps = metrics_list[0]['step'].iloc[:min_len].values

            ax.plot(steps, avg_metric, label=f'{param_name}={param_val}', color=color, linewidth=2)
            ax.fill_between(steps, avg_metric - std_metric, avg_metric + std_metric,
                           alpha=0.2, color=color)
        else:
            metrics = metrics_list[0]
            ax.plot(metrics['step'], metrics[metric_name],
                   label=f'{param_name}={param_val}', color=color, linewidth=2)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric_name.replace("_", " ").title()} Training Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if metric_name in ['format_acc', 'task_acc', 'mule_acc']:
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved trajectory comparison to {save_path}")
    plt.close()


def create_heatmap_comparison(experiments, param1, param2, metric_name, save_path):
    """Create a 2D heatmap comparing two parameters."""
    # Build a grid
    data_grid = defaultdict(dict)

    for exp in experiments:
        config = exp['config']
        if param1 not in config or param2 not in config:
            continue

        val1 = config[param1]
        val2 = config[param2]

        final_metrics = compute_final_metrics(exp['metrics'])
        data_grid[val1][val2] = final_metrics[metric_name]

    if not data_grid:
        print(f"No data found for parameters: {param1}, {param2}")
        return

    # Convert to DataFrame for heatmap
    df = pd.DataFrame(data_grid).T.sort_index()
    df = df[sorted(df.columns)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                cbar_kws={'label': metric_name.replace('_', ' ').title()})

    ax.set_xlabel(param2, fontsize=12)
    ax.set_ylabel(param1, fontsize=12)
    ax.set_title(f'{metric_name.replace("_", " ").title()} Heatmap: {param1} vs {param2}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {save_path}")
    plt.close()


def create_summary_table(experiments, output_path):
    """Create a summary table of all experiments."""
    rows = []

    for exp in experiments:
        config = exp['config']
        final_metrics = compute_final_metrics(exp['metrics'])

        row = {
            'Experiment': exp['name'],
            'MULE_REFRESH_RATE': config.get('MULE_REFRESH_RATE', 'N/A'),
            'GEN_TEMP': config.get('GEN_TEMP', 'N/A'),
            'BETA_KL': config.get('BETA_KL', 'N/A'),
            'Task_Acc': f"{final_metrics['task_acc']:.3f}",
            'Mule_Acc': f"{final_metrics['mule_acc']:.3f}",
            'Obfuscation': f"{final_metrics['obfuscation_success']:.3f}",
            'Avg_Reward': f"{final_metrics['avg_reward']:.3f}"
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_path.parent / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")

    # Create a visualization
    fig, ax = plt.subplots(figsize=(16, len(df) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Parameter Sweep Summary Table', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary table visualization to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare parameter sweep results")
    parser.add_argument("sweep_dir", type=str, help="Path to sweep results directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for comparisons (default: sweep_dir/comparisons)")
    args = parser.parse_args()

    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: Sweep directory not found: {args.sweep_dir}")
        return

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = sweep_path / "comparisons"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading sweep data from {sweep_path}")
    sweep_config, summary, experiments = load_sweep_data(sweep_path)

    if not experiments:
        print("No experiments found!")
        return

    print(f"Found {len(experiments)} experiments")

    # Determine what parameters were swept
    all_params = set()
    for exp in experiments:
        all_params.update(exp['config'].keys())

    swept_params = []
    for param in ['MULE_REFRESH_RATE', 'GEN_TEMP', 'BETA_KL']:
        if param in all_params:
            # Check if this parameter varies
            values = set(exp['config'].get(param) for exp in experiments)
            if len(values) > 1:
                swept_params.append(param)

    print(f"Swept parameters: {swept_params}")

    # Create summary table
    print("\nGenerating summary table...")
    create_summary_table(experiments, output_dir / "summary_table.png")

    # Create parameter-specific comparisons
    for param in swept_params:
        print(f"\nGenerating comparisons for {param}...")
        plot_parameter_sweep_comparison(experiments, param, output_dir / f"{param}_comparison.png")

        # Training trajectories for key metrics
        for metric in ['task_acc', 'mule_acc', 'avg_reward']:
            plot_training_trajectories(experiments, param, metric,
                                      output_dir / f"{param}_{metric}_trajectories.png")

    # If we have 2D sweeps, create heatmaps
    if len(swept_params) >= 2:
        print(f"\nGenerating heatmaps for parameter pairs...")
        for i, param1 in enumerate(swept_params):
            for param2 in swept_params[i+1:]:
                for metric in ['task_acc', 'mule_acc', 'obfuscation_success']:
                    create_heatmap_comparison(experiments, param1, param2, metric,
                                            output_dir / f"heatmap_{param1}_{param2}_{metric}.png")

    print(f"\n✅ All comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()

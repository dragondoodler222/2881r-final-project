#!/usr/bin/env python3
"""
Visualize Individual Parameter Sweep Results

This script creates detailed visualizations for individual experiment runs
within a parameter sweep, showing training dynamics and metrics over time.
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_experiment_data(exp_dir):
    """Load training metrics and config for a single experiment."""
    exp_path = Path(exp_dir)

    # Load config
    config_file = exp_path / "config.json"
    if not config_file.exists():
        print(f"Warning: No config.json found in {exp_dir}")
        config = {}
    else:
        with open(config_file) as f:
            config = json.load(f)

    # Load training metrics
    metrics_file = exp_path / "training_metrics.csv"
    if not metrics_file.exists():
        print(f"Error: No training_metrics.csv found in {exp_dir}")
        return None, None

    metrics = pd.read_csv(metrics_file)

    # Load metadata
    metadata_file = exp_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return metrics, config, metadata


def plot_training_curves(metrics, config, save_path):
    """Plot training curves showing all metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Format accuracy (should stay high)
    ax = axes[0, 0]
    ax.plot(metrics['step'], metrics['format_acc'], color='green', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Format Accuracy')
    ax.set_title('Format Compliance Over Time')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Task accuracy (correctness)
    ax = axes[0, 1]
    ax.plot(metrics['step'], metrics['task_acc'], color='blue', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Task Accuracy')
    ax.set_title('Generator Correctness Over Time')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mule accuracy (detection rate - want this LOW)
    ax = axes[1, 0]
    ax.plot(metrics['step'], metrics['mule_acc'], color='red', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mule Accuracy')
    ax.set_title('Mule Detection Rate Over Time (Lower = Better Obfuscation)')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Average reward
    ax = axes[1, 1]
    ax.plot(metrics['step'], metrics['avg_reward'], color='purple', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward Over Time')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Add config info as title
    param_str = f"MRR={config.get('MULE_REFRESH_RATE', '?')}, GT={config.get('GEN_TEMP', '?')}, BKL={config.get('BETA_KL', '?')}"
    fig.suptitle(f'Training Dynamics - {param_str}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {save_path}")
    plt.close()


def plot_adversarial_game(metrics, config, save_path):
    """Plot the adversarial game: Generator vs Mule."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both accuracies
    ax.plot(metrics['step'], metrics['task_acc'], label='Generator Correctness',
            color='blue', linewidth=2.5)
    ax.plot(metrics['step'], metrics['mule_acc'], label='Mule Detection Rate',
            color='red', linewidth=2.5)

    # Calculate obfuscation success (correct AND undetected)
    obfuscation_success = metrics['task_acc'] * (1 - metrics['mule_acc'])
    ax.plot(metrics['step'], obfuscation_success, label='Obfuscation Success (Correct & Undetected)',
            color='green', linewidth=2.5, linestyle='--')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Adversarial Game Dynamics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    param_str = f"MRR={config.get('MULE_REFRESH_RATE', '?')}, GT={config.get('GEN_TEMP', '?')}, BKL={config.get('BETA_KL', '?')}"
    ax.text(0.02, 0.98, param_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved adversarial game plot to {save_path}")
    plt.close()


def plot_reward_decomposition(metrics, config, save_path):
    """Plot reward signal with smoothing."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw reward
    ax.plot(metrics['step'], metrics['avg_reward'], alpha=0.3, color='purple', label='Raw Reward')

    # Smoothed reward (rolling average)
    window_size = min(50, len(metrics) // 10)
    if window_size > 1:
        smoothed = metrics['avg_reward'].rolling(window=window_size, center=True).mean()
        ax.plot(metrics['step'], smoothed, color='purple', linewidth=2.5, label=f'Smoothed (window={window_size})')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Reward Signal Over Time', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    param_str = f"MRR={config.get('MULE_REFRESH_RATE', '?')}, GT={config.get('GEN_TEMP', '?')}, BKL={config.get('BETA_KL', '?')}"
    ax.text(0.02, 0.98, param_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward decomposition to {save_path}")
    plt.close()


def plot_summary_stats(metrics, config, save_path):
    """Create a summary statistics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Final performance (last 20% of training)
    cutoff = int(len(metrics) * 0.8)
    final_metrics = metrics.iloc[cutoff:]

    final_stats = {
        'Format Acc': final_metrics['format_acc'].mean(),
        'Task Acc': final_metrics['task_acc'].mean(),
        'Mule Acc\n(lower=better)': final_metrics['mule_acc'].mean(),
        'Obfuscation\nSuccess': (final_metrics['task_acc'] * (1 - final_metrics['mule_acc'])).mean()
    }

    # Bar plot of final stats
    ax = axes[0]
    colors = ['green', 'blue', 'red', 'purple']
    bars = ax.bar(range(len(final_stats)), list(final_stats.values()), color=colors, alpha=0.7)
    ax.set_xticks(range(len(final_stats)))
    ax.set_xticklabels(list(final_stats.keys()), fontsize=10)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Final Performance (Last 20% of Training)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # Trajectory visualization (simplified)
    ax = axes[1]
    # Bin the data into 10 segments
    n_bins = 10
    bin_size = len(metrics) // n_bins
    binned_task = [metrics['task_acc'].iloc[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]
    binned_mule = [metrics['mule_acc'].iloc[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)]

    x_pos = np.arange(n_bins)
    width = 0.35

    ax.bar(x_pos - width/2, binned_task, width, label='Task Acc', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, binned_mule, width, label='Mule Acc', color='red', alpha=0.7)
    ax.set_xlabel('Training Progress (Binned)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training Progression', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{int((i+1)*100/n_bins)}%' for i in range(n_bins)], fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    param_str = f"MRR={config.get('MULE_REFRESH_RATE', '?')}, GT={config.get('GEN_TEMP', '?')}, BKL={config.get('BETA_KL', '?')}"
    fig.suptitle(f'Summary Statistics - {param_str}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary statistics to {save_path}")
    plt.close()


def visualize_experiment(exp_dir, output_dir=None):
    """Create all visualizations for a single experiment."""
    exp_path = Path(exp_dir)

    # Load data
    result = load_experiment_data(exp_path)
    if result is None or result[0] is None:
        print(f"Failed to load data from {exp_dir}")
        return

    metrics, config, metadata = result

    # Setup output directory
    if output_dir is None:
        output_dir = exp_path / "visualizations"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations for: {exp_path.name}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Generate plots
    plot_training_curves(metrics, config, output_dir / "training_curves.png")
    plot_adversarial_game(metrics, config, output_dir / "adversarial_game.png")
    plot_reward_decomposition(metrics, config, output_dir / "reward_signal.png")
    plot_summary_stats(metrics, config, output_dir / "summary_stats.png")

    print(f"\n✅ All visualizations saved to {output_dir}")


def visualize_all_experiments(sweep_dir):
    """Create visualizations for all experiments in a sweep."""
    sweep_path = Path(sweep_dir)

    # Find all experiment directories
    exp_dirs = [d for d in sweep_path.iterdir() if d.is_dir() and (d / "training_metrics.csv").exists()]

    print(f"Found {len(exp_dirs)} experiments to visualize")

    for exp_dir in exp_dirs:
        try:
            visualize_experiment(exp_dir)
        except Exception as e:
            print(f"Error visualizing {exp_dir.name}: {e}")

    print(f"\n✅ Finished visualizing all experiments")


def main():
    parser = argparse.ArgumentParser(description="Visualize parameter sweep results")
    parser.add_argument("sweep_dir", type=str, help="Path to sweep results directory")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Specific experiment name to visualize (if not provided, visualizes all)")
    args = parser.parse_args()

    sweep_path = Path(args.sweep_dir)
    if not sweep_path.exists():
        print(f"Error: Sweep directory not found: {args.sweep_dir}")
        return

    if args.exp_name:
        exp_path = sweep_path / args.exp_name
        if not exp_path.exists():
            print(f"Error: Experiment not found: {args.exp_name}")
            return
        visualize_experiment(exp_path)
    else:
        visualize_all_experiments(sweep_path)


if __name__ == "__main__":
    main()

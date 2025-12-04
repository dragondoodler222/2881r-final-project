#!/usr/bin/env python3
"""
Visualize training metrics as separate plots without phase separation.
Saves each metric as its own PNG file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize training metrics')
parser.add_argument('csv_files', nargs='+', help='One or more CSV files to visualize')
parser.add_argument('--max_step', type=int, default=None, help='Maximum step to include (default: no limit)')
parser.add_argument('--output_prefix', type=str, default='', help='Prefix for output filenames')
args = parser.parse_args()

# Read CSV files
print("Loading training data...")
dataframes = []
for i, csv_file in enumerate(args.csv_files):
    try:
        df = pd.read_csv(csv_file)
        print(f"  Loaded: {csv_file} ({len(df)} steps)")

        # If multiple files, adjust subsequent steps to continue from previous
        if i > 0 and len(dataframes) > 0:
            offset = dataframes[-1]['step'].max()
            df_adjusted = df.copy()
            df_adjusted['step'] = df['step'] + offset
            dataframes.append(df_adjusted)
        else:
            dataframes.append(df)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        sys.exit(1)

# Combine datasets if multiple files
if len(dataframes) > 1:
    combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined: {len(combined)} total steps")
else:
    combined = dataframes[0]

# Filter to max step if specified
if args.max_step is not None:
    combined = combined[combined['step'] <= args.max_step]
    print(f"Filtered to steps 1-{combined['step'].max()}")
else:
    print(f"Total steps: {len(combined)} (1-{combined['step'].max()})")

# Color scheme
color = '#2E86AB'  # Single color for unified view

# Helper function for output filenames
def output_filename(name):
    if args.output_prefix:
        return f"{args.output_prefix}_{name}"
    return name

# Plot 1: Format Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined['step'], combined['format_acc'], color=color, linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Format Accuracy', fontsize=12)
ax.set_title('Format Accuracy Over Training', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
filename = output_filename('format_accuracy.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 2: Task Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined['step'], combined['task_acc'], color='#06A77D', linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Task Accuracy', fontsize=12)
ax.set_title('Task Accuracy Over Training (Generator Correctness)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
filename = output_filename('task_accuracy.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 3: Mule Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined['step'], combined['mule_acc'], color='#D62246', linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Adversary Accuracy', fontsize=12)
ax.set_title('Aversary Accuracy Over Training (Prediction Ability)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
filename = output_filename('mule_accuracy.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 4: Average Reward
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined['step'], combined['avg_reward'], color='#F18F01', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Average Reward', fontsize=12)
ax.set_title('Average Reward Over Training', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
filename = output_filename('average_reward.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

# Plot 5: Adversarial Dynamics (combined view)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(combined['step'], combined['task_acc'], linewidth=2.5, label='Generator Task Accuracy', color='#06A77D')
ax.plot(combined['step'], combined['mule_acc'], linewidth=2.5, label='Adversary Prediction Accuracy', color='#D62246')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Adversarial Dynamics: Generator vs Adversary', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
plt.tight_layout()
filename = output_filename('adversarial_dynamics_combined.png')
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
plt.close()

print("\n✓ All individual plots saved successfully!")
print("\nGenerated files:")
for name in ['format_accuracy.png', 'task_accuracy.png', 'mule_accuracy.png',
             'average_reward.png', 'adversarial_dynamics_combined.png']:
    print(f"  - {output_filename(name)}")

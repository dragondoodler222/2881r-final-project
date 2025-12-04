#!/usr/bin/env python3
"""
Visualize training metrics as separate plots without phase separation.
Saves each metric as its own PNG file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
print("Loading training data...")
phase1 = pd.read_csv('training_metrics_31-8B.csv')
phase2 = pd.read_csv('training_metrics_20251203_044833.csv')

# Adjust phase 2 steps to continue from phase 1
phase2_adjusted = phase2.copy()
phase2_adjusted['step'] = phase2['step'] + phase1['step'].max()

# Combine datasets
combined = pd.concat([phase1, phase2_adjusted], ignore_index=True)

# Filter to only include up to step 1500
combined = combined[combined['step'] <= 1500]
print(f"Combined: {len(combined)} total steps (1-{combined['step'].max()})")

# Color scheme
color = '#2E86AB'  # Single color for unified view

# Plot 1: Format Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(combined['step'], combined['format_acc'], color=color, linewidth=2)
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Format Accuracy', fontsize=12)
ax.set_title('Format Accuracy Over Training', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig('format_accuracy.png', dpi=300, bbox_inches='tight')
print("Saved: format_accuracy.png")
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
plt.savefig('task_accuracy.png', dpi=300, bbox_inches='tight')
print("Saved: task_accuracy.png")
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
plt.savefig('mule_accuracy.png', dpi=300, bbox_inches='tight')
print("Saved: mule_accuracy.png")
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
plt.savefig('average_reward.png', dpi=300, bbox_inches='tight')
print("Saved: average_reward.png")
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
plt.savefig('adversarial_dynamics_combined.png', dpi=300, bbox_inches='tight')
print("Saved: adversarial_dynamics_combined.png")
plt.close()

print("\n✓ All individual plots saved successfully!")
print("\nGenerated files:")
print("  - format_accuracy.png")
print("  - task_accuracy.png")
print("  - mule_accuracy.png")
print("  - average_reward.png")
print("  - adversarial_dynamics_combined.png")

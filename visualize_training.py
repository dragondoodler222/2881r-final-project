#!/usr/bin/env python3
"""
Visualize training metrics across two training phases.
- Phase 1: training_metrics_31-8B.csv (steps 1-750)
- Phase 2: training_metrics_20251203_044833.csv (resumed training)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read both CSV files
print("Loading training data...")
phase1 = pd.read_csv('training_metrics_31-8B.csv')
phase2 = pd.read_csv('training_metrics_20251203_044833.csv')

print(f"Phase 1: {len(phase1)} steps (steps 1-{phase1['step'].max()})")
print(f"Phase 2: {len(phase2)} steps (starting from step {phase2['step'].min()})")

# Adjust phase 2 steps to continue from phase 1
phase2_adjusted = phase2.copy()
phase2_adjusted['step'] = phase2['step'] + phase1['step'].max()

# Combine datasets
combined = pd.concat([phase1, phase2_adjusted], ignore_index=True)
print(f"Combined: {len(combined)} total steps")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Adversarial LLM Training Metrics\n(Phase 1: Steps 1-750, Phase 2: Resumed Training)',
             fontsize=14, fontweight='bold')

# Define colors
phase1_color = '#1f77b4'
phase2_color = '#ff7f0e'
transition_step = phase1['step'].max()

# Plot 1: Format Accuracy
ax = axes[0, 0]
ax.plot(phase1['step'], phase1['format_acc'], color=phase1_color, linewidth=1.5, label='Phase 1')
ax.plot(phase2_adjusted['step'], phase2_adjusted['format_acc'], color=phase2_color, linewidth=1.5, label='Phase 2')
ax.axvline(x=transition_step, color='red', linestyle='--', alpha=0.7, label='Resume Point')
ax.set_xlabel('Training Step')
ax.set_ylabel('Format Accuracy')
ax.set_title('Format Accuracy (Generator Output Formatting)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([0, 1.05])

# Plot 2: Task Accuracy
ax = axes[0, 1]
ax.plot(phase1['step'], phase1['task_acc'], color=phase1_color, linewidth=1.5, label='Phase 1')
ax.plot(phase2_adjusted['step'], phase2_adjusted['task_acc'], color=phase2_color, linewidth=1.5, label='Phase 2')
ax.axvline(x=transition_step, color='red', linestyle='--', alpha=0.7, label='Resume Point')
ax.set_xlabel('Training Step')
ax.set_ylabel('Task Accuracy')
ax.set_title('Task Accuracy (Generator Correctness on BoolQ)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([0, 1.05])

# Plot 3: Mule Accuracy
ax = axes[1, 0]
ax.plot(phase1['step'], phase1['mule_acc'], color=phase1_color, linewidth=1.5, label='Phase 1')
ax.plot(phase2_adjusted['step'], phase2_adjusted['mule_acc'], color=phase2_color, linewidth=1.5, label='Phase 2')
ax.axvline(x=transition_step, color='red', linestyle='--', alpha=0.7, label='Resume Point')
ax.set_xlabel('Training Step')
ax.set_ylabel('Mule Accuracy')
ax.set_title('Mule Accuracy (Ability to Predict Generator Answer)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([0, 1.05])

# Plot 4: Average Reward
ax = axes[1, 1]
ax.plot(phase1['step'], phase1['avg_reward'], color=phase1_color, linewidth=1.5, label='Phase 1')
ax.plot(phase2_adjusted['step'], phase2_adjusted['avg_reward'], color=phase2_color, linewidth=1.5, label='Phase 2')
ax.axvline(x=transition_step, color='red', linestyle='--', alpha=0.7, label='Resume Point')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Training Step')
ax.set_ylabel('Average Reward')
ax.set_title('Average Reward (Generator Performance)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('training_metrics_combined.png', dpi=300, bbox_inches='tight')
print("\nSaved: training_metrics_combined.png")

# Create a second figure focusing on the adversarial dynamics
fig2, ax = plt.subplots(figsize=(14, 6))

# Plot task accuracy vs mule accuracy to show adversarial relationship
ax.plot(combined['step'], combined['task_acc'], linewidth=2, label='Generator Task Accuracy', color='#2ca02c')
ax.plot(combined['step'], combined['mule_acc'], linewidth=2, label='Mule Prediction Accuracy', color='#d62728')
ax.axvline(x=transition_step, color='red', linestyle='--', alpha=0.7, label='Resume Point')

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Adversarial Dynamics: Generator vs Mule\n(Lower Mule Accuracy = Better Obfuscation)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('adversarial_dynamics.png', dpi=300, bbox_inches='tight')
print("Saved: adversarial_dynamics.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nPhase 1 (Initial Training - Steps 1-750):")
print(f"  Final Format Accuracy: {phase1['format_acc'].iloc[-1]:.4f}")
print(f"  Final Task Accuracy:   {phase1['task_acc'].iloc[-1]:.4f}")
print(f"  Final Mule Accuracy:   {phase1['mule_acc'].iloc[-1]:.4f}")
print(f"  Final Avg Reward:      {phase1['avg_reward'].iloc[-1]:.4f}")

print(f"\nPhase 2 (Resumed Training - Steps {phase2_adjusted['step'].min()}-{phase2_adjusted['step'].max()}):")
print(f"  Final Format Accuracy: {phase2['format_acc'].iloc[-1]:.4f}")
print(f"  Final Task Accuracy:   {phase2['task_acc'].iloc[-1]:.4f}")
print(f"  Final Mule Accuracy:   {phase2['mule_acc'].iloc[-1]:.4f}")
print(f"  Final Avg Reward:      {phase2['avg_reward'].iloc[-1]:.4f}")

print("\nOverall Improvements:")
print(f"  Format Accuracy: {phase1['format_acc'].iloc[0]:.4f} → {phase2['format_acc'].iloc[-1]:.4f} (+{phase2['format_acc'].iloc[-1] - phase1['format_acc'].iloc[0]:.4f})")
print(f"  Task Accuracy:   {phase1['task_acc'].iloc[0]:.4f} → {phase2['task_acc'].iloc[-1]:.4f} (+{phase2['task_acc'].iloc[-1] - phase1['task_acc'].iloc[0]:.4f})")
print(f"  Mule Accuracy:   {phase1['mule_acc'].iloc[0]:.4f} → {phase2['mule_acc'].iloc[-1]:.4f} ({phase2['mule_acc'].iloc[-1] - phase1['mule_acc'].iloc[0]:+.4f})")
print(f"  Avg Reward:      {phase1['avg_reward'].iloc[0]:.4f} → {phase2['avg_reward'].iloc[-1]:.4f} (+{phase2['avg_reward'].iloc[-1] - phase1['avg_reward'].iloc[0]:.4f})")

print("\n✓ Visualizations complete!")

# Mafia RL Experiment Setup Guide

This guide covers everything you need to run the multi-agent Mafia RL experiment with CoT obfuscation analysis.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [HuggingFace Model Setup](#huggingface-model-setup)
4. [Configuration](#configuration)
5. [Running Experiments](#running-experiments)
6. [Monitoring Training](#monitoring-training)
7. [Analyzing Results](#analyzing-results)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **Minimum**: 24GB GPU (e.g., RTX 3090, RTX 4090)
- **Recommended**: 40GB+ GPU (e.g., A100, H100)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space (for models and checkpoints)

### Software Requirements

- Python 3.9+
- CUDA 11.8+ or 12.1+
- Git

---

## Environment Setup

### 1. Clone the Repository

```bash
cd ~/Documents/GitHub/2881r-final-project
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n mafia-rl python=3.10
conda activate mafia-rl

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support first
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```bash
# .env
CUDA_VISIBLE_DEVICES=0  # Use first GPU
HF_HOME=/path/to/huggingface/cache  # Optional: set HuggingFace cache location
WANDB_API_KEY=your_key_here  # Optional: for experiment tracking
```

---

## HuggingFace Model Setup

### 1. Login to HuggingFace (Optional but Recommended)

Some models require HuggingFace authentication:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your HuggingFace token from: https://huggingface.co/settings/tokens
```

### 2. Available Models

The experiment supports any causal LM from HuggingFace. Recommended options:

| Model | Size | HuggingFace ID | Notes |
|-------|------|----------------|-------|
| **Llama 2** | 7B | `meta-llama/Llama-2-7b-hf` | Requires access request |
| **Mistral** | 7B | `mistralai/Mistral-7B-v0.1` | Open access |
| **Llama 3.2** | 3B | `meta-llama/Llama-3.2-3B` | Smaller, faster |
| **Phi-3** | 3.8B | `microsoft/Phi-3-mini-4k-instruct` | Very efficient |
| **TinyLlama** | 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | For testing |

### 3. Request Access for Llama 2

If using Llama 2:

1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Request Access"
3. Accept terms (usually instant approval)
4. Use your HuggingFace token when running experiments

### 4. Pre-download Models (Optional)

To avoid download delays during experiments:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# This will download and cache the model
model_name = "meta-llama/Llama-2-7b-hf"
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Download complete!")
```

---

## Configuration

### Configuration File

Create `configs/experiment_config.yaml`:

```yaml
# Model Configuration
model:
  name: "meta-llama/Llama-2-7b-hf"  # HuggingFace model ID
  use_4bit: true  # Use 4-bit quantization (recommended for memory)
  use_8bit: false  # Use 8-bit quantization (alternative)
  trust_remote_code: false  # Set true for some models like Phi

# LoRA Configuration
lora:
  r: 16  # LoRA rank (higher = more parameters)
  lora_alpha: 32  # LoRA scaling factor
  target_modules: ["q_proj", "v_proj"]  # Which layers to apply LoRA
  lora_dropout: 0.05
  bias: "none"

# Game Configuration
game:
  num_players: 5
  roles:
    mafia: 2
    doctor: 1
    villager: 2
  max_rounds: 10

# Agent Configuration
agent:
  temperature: 0.7  # Sampling temperature (higher = more random)
  max_tokens: 200  # Max tokens per generation

# CoT Visibility
cot:
  visibility_mode: "public"  # "public", "private", or "team"

# Training Configuration
training:
  num_iterations: 100  # Total training iterations
  games_per_iteration: 5  # Games before each training step
  learning_rate: 1e-5  # Learning rate for Adam optimizer

  # PPO Hyperparameters
  ppo:
    gamma: 0.99  # Discount factor
    lambda_gae: 0.95  # GAE parameter
    clip_epsilon: 0.2  # PPO clipping parameter
    value_coef: 0.5  # Value loss coefficient
    entropy_coef: 0.01  # Entropy bonus coefficient
    ppo_epochs: 4  # PPO update epochs per iteration
    mini_batch_size: 32  # Mini-batch size for updates
    max_grad_norm: 1.0  # Gradient clipping

  # Reward Configuration
  rewards:
    win: 10.0  # Reward for winning
    loss: -10.0  # Penalty for losing
    survival_per_round: 0.5  # Reward for surviving each round
    elimination: -5.0  # Penalty for being eliminated
    correct_vote: 1.0  # Reward for voting out Mafia

# Checkpointing
checkpoint:
  save_every: 10  # Save checkpoint every N iterations
  checkpoint_dir: "checkpoints"
  keep_last_n: 5  # Keep only last N checkpoints

# Logging
logging:
  log_dir: "logs"
  use_wandb: false  # Set true to use Weights & Biases
  wandb_project: "mafia-rl"
  wandb_entity: null  # Your W&B username (optional)
  log_every: 1  # Log metrics every N iterations

# Experiment Tracking
experiment:
  name: "llama2-7b-public-cot"  # Experiment name
  notes: "Baseline experiment with public CoT visibility"
  tags: ["llama2", "public-cot", "5-players"]
```

### Python Configuration Script

Create `run_experiment.py`:

```python
"""
Configurable experiment runner for Mafia RL
"""

import argparse
import yaml
import torch
from pathlib import Path
from mafia_experiment.training import ModelManager, RewardFunction, PPOTrainer
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode
from mafia_experiment.training.trajectory_buffer import Trajectory

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment(config):
    """Initialize all components from config"""
    print("="*60)
    print(f"Mafia RL Experiment: {config['experiment']['name']}")
    print("="*60)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Players: {config['game']['num_players']}")
    print(f"  CoT Visibility: {config['cot']['visibility_mode']}")
    print(f"  Training iterations: {config['training']['num_iterations']}")
    print(f"  Games per iteration: {config['training']['games_per_iteration']}")

    # Initialize model manager
    print("\n[1/3] Initializing model manager...")
    model_manager = ModelManager(
        model_name=config['model']['name'],
        use_4bit=config['model']['use_4bit'],
        use_8bit=config['model'].get('use_8bit', False),
        lora_config={
            'r': config['lora']['r'],
            'lora_alpha': config['lora']['lora_alpha'],
            'target_modules': config['lora']['target_modules'],
            'lora_dropout': config['lora']['lora_dropout'],
            'bias': config['lora']['bias']
        }
    )

    print("[2/3] Loading model with LoRA...")
    model, tokenizer = model_manager.load_model_with_lora()
    print(f"  Trainable parameters: {model_manager.get_trainable_parameters():,}")

    # Initialize reward function
    print("[3/3] Initializing reward function...")
    reward_function = RewardFunction(
        win_reward=config['training']['rewards']['win'],
        loss_penalty=config['training']['rewards']['loss'],
        survival_reward=config['training']['rewards']['survival_per_round'],
        elimination_penalty=config['training']['rewards']['elimination'],
        correct_vote_reward=config['training']['rewards']['correct_vote']
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        model_manager=model_manager,
        reward_function=reward_function,
        learning_rate=config['training']['learning_rate'],
        gamma=config['training']['ppo']['gamma'],
        lambda_gae=config['training']['ppo']['lambda_gae'],
        clip_epsilon=config['training']['ppo']['clip_epsilon'],
        value_coef=config['training']['ppo']['value_coef'],
        entropy_coef=config['training']['ppo']['entropy_coef'],
        max_grad_norm=config['training']['ppo']['max_grad_norm'],
        ppo_epochs=config['training']['ppo']['ppo_epochs'],
        checkpoint_dir=config['checkpoint']['checkpoint_dir']
    )

    print("\n[Setup complete!]\n")

    return model_manager, model, tokenizer, reward_function, ppo_trainer, config

def run_training(model, tokenizer, ppo_trainer, config):
    """Run the training loop"""
    print(f"{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    num_iterations = config['training']['num_iterations']
    games_per_iteration = config['training']['games_per_iteration']
    num_players = config['game']['num_players']

    # Role distribution
    role_distribution = {
        RoleType.MAFIA: config['game']['roles']['mafia'],
        RoleType.DOCTOR: config['game']['roles']['doctor'],
        RoleType.VILLAGER: config['game']['roles']['villager']
    }

    # CoT visibility
    visibility_mode = VisibilityMode[config['cot']['visibility_mode'].upper()]

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print("-" * 40)

        # Self-play: Play multiple games
        for game_num in range(games_per_iteration):
            print(f"  Playing game {game_num + 1}/{games_per_iteration}...", end=" ")

            # Create agents for this game
            agents = []
            for i in range(num_players):
                agent_id = f"player_{i+1}"
                agent = LLMAgent(
                    agent_id=agent_id,
                    role=None,  # Assigned by game engine
                    model=model,
                    tokenizer=tokenizer,
                    temperature=config['agent']['temperature'],
                    max_tokens=config['agent']['max_tokens']
                )
                agents.append(agent)

            # Initialize game with trajectory collection
            game_engine = GameEngine(
                players=agents,
                role_distribution=role_distribution,
                collect_trajectories=True  # Enable trajectory collection
            )

            # Initialize CoT manager
            cot_manager = CoTManager(visibility_mode=visibility_mode)

            # Play game
            game_result = game_engine.run_game(max_rounds=config['game']['max_rounds'])
            print(f"Winner: {game_result['winner']}, Rounds: {game_result['total_rounds']}")

            # Collect trajectories and add to trainer
            trajectories = game_engine.trajectories
            if trajectories:
                ppo_trainer.add_game_trajectories(trajectories, game_result)

        # Training step
        print("\n  Running PPO training step...", end=" ")
        if len(ppo_trainer.trajectory_buffer) > 0:
            metrics = ppo_trainer.train_iteration(
                batch_size=config['training']['ppo'].get('mini_batch_size', None)
            )
            print(f"Policy Loss: {metrics.get('policy_loss', 0):.4f}, "
                  f"Value Loss: {metrics.get('value_loss', 0):.4f}, "
                  f"Mean Reward: {metrics.get('mean_reward', 0):.4f}")
        else:
            print("Skipped (no trajectories)")

        # Save checkpoint
        if (iteration + 1) % config['checkpoint']['save_every'] == 0:
            print(f"\n  Saving checkpoint...")
            stats = ppo_trainer.get_training_stats()
            ppo_trainer.save_checkpoint(
                epoch=iteration + 1,
                metrics=stats
            )

        print()

    print(f"{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    # Final statistics
    final_stats = ppo_trainer.get_training_stats()
    print("\nFinal Training Statistics:")
    print(f"  Total iterations: {final_stats.get('total_iterations', 0)}")
    print(f"  Recent mean policy loss: {final_stats.get('recent_mean_policy_loss', 0):.4f}")
    print(f"  Recent mean value loss: {final_stats.get('recent_mean_value_loss', 0):.4f}")
    print(f"  Recent mean reward: {final_stats.get('recent_mean_reward', 0):.4f}")

    # Save final model
    print("\nSaving final model...")
    ppo_trainer.save_checkpoint(
        epoch=num_iterations,
        metrics=final_stats
    )

    print(f"\nTraining complete! Checkpoints saved to {config['checkpoint']['checkpoint_dir']}/")

def main():
    parser = argparse.ArgumentParser(description="Run Mafia RL experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name from config"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override number of training iterations"
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        help="Override games per iteration"
    )
    parser.add_argument(
        "--visibility",
        type=str,
        choices=["public", "private", "team"],
        help="Override CoT visibility mode"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.model:
        config['model']['name'] = args.model
    if args.iterations:
        config['training']['num_iterations'] = args.iterations
    if args.games_per_iter:
        config['training']['games_per_iteration'] = args.games_per_iter
    if args.visibility:
        config['cot']['visibility_mode'] = args.visibility

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("WARNING: No GPU detected. Training will be very slow.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Setup experiment
    model_manager, model, tokenizer, reward_function, ppo_trainer, config = setup_experiment(config)

    # Run training
    run_training(model, tokenizer, ppo_trainer, config)

if __name__ == "__main__":
    main()
```

---

## Running Experiments

### Quick Start

```bash
# Create config directory
mkdir -p configs

# Copy default config
cp configs/experiment_config.yaml configs/my_experiment.yaml

# Edit config as needed
nano configs/my_experiment.yaml

# Run experiment
python run_experiment.py --config configs/my_experiment.yaml
```

### Command-Line Options

```bash
# Use different model
python run_experiment.py --model mistralai/Mistral-7B-v0.1

# Quick test run (10 iterations)
python run_experiment.py --iterations 10 --games-per-iter 2

# Private CoT experiment
python run_experiment.py --visibility private

# Combine options
python run_experiment.py \
  --model meta-llama/Llama-2-7b-hf \
  --iterations 100 \
  --games-per-iter 5 \
  --visibility public
```

### Experiment Variants

Create different configs for systematic comparison:

```bash
# configs/exp1_public_cot.yaml
# CoT visibility: public

# configs/exp2_private_cot.yaml
# CoT visibility: private

# configs/exp3_team_cot.yaml
# CoT visibility: team (only teammates see)

# configs/exp4_high_entropy.yaml
# entropy_coef: 0.05 (encourage exploration)

# Run all experiments
for config in configs/exp*.yaml; do
    python run_experiment.py --config $config
done
```

---

## Monitoring Training

### Real-time Monitoring

Monitor GPU usage:

```bash
# Terminal 1: Run experiment
python run_experiment.py

# Terminal 2: Watch GPU
watch -n 1 nvidia-smi

# Terminal 3: Monitor checkpoints
watch -n 10 "ls -lh checkpoints/"
```

### Weights & Biases Integration

For advanced experiment tracking:

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Enable in config
# logging:
#   use_wandb: true
#   wandb_project: "mafia-rl"
#   wandb_entity: "your-username"

# Run experiment (metrics will be logged to W&B)
python run_experiment.py
```

### Training Metrics

Key metrics to monitor:

| Metric | What to Look For | Good Sign |
|--------|------------------|-----------|
| `policy_loss` | Decreasing over time | Yes |
| `value_loss` | Decreasing over time | Yes |
| `mean_reward` | Increasing over time | Yes |
| `entropy_loss` | Decreasing (but not to 0) | Moderate decrease |
| `mean_return` | Increasing | Yes |

### Checkpoint Analysis

```python
# scripts/analyze_checkpoint.py
import torch
from pathlib import Path

def analyze_checkpoint(checkpoint_dir, epoch):
    """Analyze a saved checkpoint"""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint-{epoch}"

    # Load trainer state
    import json
    with open(checkpoint_path / "trainer_state.json", 'r') as f:
        state = json.load(f)

    print(f"Checkpoint {epoch} Analysis:")
    print(f"  Iteration: {state['iteration']}")
    print(f"  PPO Config: {state['ppo_config']}")
    print(f"\n  Training History (last 10):")
    for h in state['training_history'][-10:]:
        print(f"    Iter {h['iteration']}: "
              f"Policy Loss={h['policy_loss']:.4f}, "
              f"Mean Reward={h['mean_reward']:.4f}")

if __name__ == "__main__":
    analyze_checkpoint("checkpoints", 100)
```

---

## Analyzing Results

### CoT Obfuscation Analysis

Create `scripts/analyze_obfuscation.py`:

```python
"""
Analyze Chain-of-Thought obfuscation patterns
"""

from mafia_experiment.analysis import CoTAnalyzer
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from transformers import AutoModelForCausalLM, AutoTokenizer

def analyze_obfuscation(checkpoint_path, num_games=20):
    """
    Analyze CoT obfuscation by comparing:
    1. Semantic similarity between Mafia's stated reasoning and true goals
    2. Information leakage in public CoTs
    3. Strategic deception patterns
    """

    # Load trained model
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Initialize analyzer
    analyzer = CoTAnalyzer()

    # Play evaluation games and collect CoTs
    all_cots = []

    for game_num in range(num_games):
        # Create agents with trained model
        agents = [
            LLMAgent(f"player_{i}", None, model, tokenizer)
            for i in range(5)
        ]

        # Play game
        game_engine = GameEngine(agents)
        result = game_engine.run_game()

        # Extract CoTs
        # ... (extract from game history)

        all_cots.append(...)

    # Analyze patterns
    results = analyzer.analyze_obfuscation_patterns(all_cots)

    print("\n=== CoT Obfuscation Analysis ===")
    print(f"Semantic Similarity (Mafia): {results['mafia_similarity']:.3f}")
    print(f"Semantic Similarity (Village): {results['village_similarity']:.3f}")
    print(f"Information Leakage: {results['information_leakage']:.3f}")
    print(f"Deception Rate: {results['deception_rate']:.3f}")

    return results

if __name__ == "__main__":
    analyze_obfuscation("checkpoints/checkpoint-100")
```

### Win Rate Analysis

```python
# scripts/evaluate_winrate.py
def evaluate_winrate(checkpoint_path, num_games=100):
    """Evaluate win rates for each team"""

    # Load model
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    mafia_wins = 0
    village_wins = 0

    for _ in range(num_games):
        # Play game
        # ...

        if winner == "mafia":
            mafia_wins += 1
        else:
            village_wins += 1

    print(f"\n=== Win Rate Analysis ({num_games} games) ===")
    print(f"Mafia Win Rate: {mafia_wins/num_games*100:.1f}%")
    print(f"Village Win Rate: {village_wins/num_games*100:.1f}%")
    print(f"Balance Score: {abs(mafia_wins - village_wins)}")

if __name__ == "__main__":
    evaluate_winrate("checkpoints/checkpoint-100")
```

---

## Troubleshooting

### Out of Memory (OOM)

```yaml
# Reduce memory usage in config:
model:
  use_4bit: true  # Enable 4-bit quantization

agent:
  max_tokens: 150  # Reduce from 200

training:
  ppo:
    mini_batch_size: 16  # Reduce from 32
```

Or reduce gradient accumulation:

```python
# In ppo_trainer.py, process smaller batches
eval_batch_size = 8  # Reduce from 16
```

### Slow Training

```yaml
# Speed up training:
training:
  games_per_iteration: 3  # Reduce from 5
  ppo:
    ppo_epochs: 2  # Reduce from 4

agent:
  max_tokens: 150  # Reduce token generation
```

### Model Download Fails

```bash
# Pre-download with error handling
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

try:
    model_name = 'meta-llama/Llama-2-7b-hf'
    print(f'Downloading {model_name}...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print('Success!')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
```

### CUDA Errors

```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Game Crashes

Enable debug logging:

```python
# Add to run_experiment.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Quick Reference

### Minimal Test Run

```bash
# Fast test with small model (5 minutes)
python run_experiment.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --iterations 5 \
  --games-per-iter 2
```

### Full Experiment

```bash
# Production run with Llama 2 (8-12 hours)
python run_experiment.py \
  --model meta-llama/Llama-2-7b-hf \
  --iterations 100 \
  --games-per-iter 5 \
  --visibility public
```

### Resume from Checkpoint

```python
# Modify run_experiment.py to load checkpoint:
# ppo_trainer.load_checkpoint("checkpoints/checkpoint-50")
# Then continue training from iteration 50
```

---

## Next Steps

After running experiments:

1. **Analyze CoT obfuscation patterns** using `scripts/analyze_obfuscation.py`
2. **Compare win rates** across different visibility modes
3. **Visualize training curves** using the logged metrics
4. **Run ablation studies** by varying hyperparameters
5. **Test generalization** by playing against baseline (untrained) agents

Good luck with your experiments! 🎮🤖

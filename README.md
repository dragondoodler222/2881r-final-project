# Multi-Agent Mafia: Obfuscation in Information Asymmetric Games

An experiment on adversarial and emergent obfuscation of interpretability using reinforcement learning in the Mafia game setting.

## Overview

This project extends a two-agent Chain-of-Thought (CoT) prediction experiment to a multi-agent information asymmetric game setting using Mafia. The key research question: **Do LLM agents learn to obfuscate their chain-of-thought reasoning when it is publicly visible in an adversarial game?**

### Key Features

- 🎮 **Full Mafia game implementation** with Mafia, Villager, and Doctor roles
- 🤖 **RL-based agent training** using PPO with LoRA (parameter-efficient fine-tuning)
- 🧠 **Chain-of-Thought (CoT) management** with configurable visibility modes
- 📊 **Obfuscation metrics** to measure deception and information leakage
- 🔬 **Experimental framework** for systematic evaluation

## Setup

### Requirements

- Python 3.9+
- CUDA-capable GPU with 16GB+ VRAM (or use 4-bit quantization for smaller GPUs)
- 50-100GB storage for models and checkpoints

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/2881r-final-project.git
cd 2881r-final-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Configure Weights & Biases for experiment tracking:
```bash
echo "WANDB_API_KEY=your_wandb_key" >> .env
```

## Project Structure

```
mafia_experiment/
├── agents/           # LLM agent implementations
├── game/             # Game engine, state, phases, and roles
├── cot/              # CoT management and analysis
├── training/         # PPO trainer, rewards, trajectory buffer
├── parallel/         # Multi-worker game generation infrastructure
├── utils/            # Prompts and utilities
└── analysis/         # Analysis tools

# Top-level scripts
train_and_evaluate.py    # Main training script
run_evaluation_games.py  # Run evaluation games with trained checkpoints
analyze_results.py       # Analyze evaluation metrics

# Output directories (created automatically)
checkpoints/          # Saved model checkpoints
logs/                 # Training logs and game traces
eval_games/           # Evaluation game outputs
```

## Quick Start

### 1. Training a Model

Run PPO training with the main training script:

```bash
python train_and_evaluate.py
```

This will:
1. Load Llama-3.2-1B-Instruct with 4-bit quantization
2. Apply LoRA adapters for efficient fine-tuning
3. Run parallel self-play games of Mafia (8 workers by default)
4. Train agents using PPO
5. Save checkpoints every iteration to `checkpoints/`

#### Training Configuration

Edit the config dict in `train_and_evaluate.py`:

```python
config = {
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "num_players": 6,
    "cot_visibility": VisibilityMode.PUBLIC,  # PUBLIC or PRIVATE
    "num_training_iterations": 64,
    "games_per_iteration": 32,
    "num_workers": 8,
    "use_4bit": True,
    "resume_checkpoint": None  # Set to "checkpoints/checkpoint-5" to resume
}
```

### 2. Running Evaluation Games

After training, evaluate your checkpoint against a baseline:

```bash
python run_evaluation_games.py
```

Or use the evaluation script with custom configuration:

```python
# In run_evaluation_games.py, modify:
config = {
    "model_type": "checkpoint",  # "base" or "checkpoint"
    "checkpoint_path": "checkpoints/checkpoint-20",
    "num_games": 100,
    "output_dir": "eval_games/my_eval"
}
```

### 3. Analyzing Results

After running evaluations, analyze the results:

```bash
python analyze_results.py
```

This outputs win rates, survival statistics, and comparison tables.

## Experiment Design

### Research Questions

1. Do Mafia agents learn to obfuscate their reasoning when CoT is public?
2. What obfuscation strategies emerge naturally through RL?

### CoT Visibility Modes

- **PUBLIC**: All players see each other's internal reasoning (creates incentive to obfuscate)
- **PRIVATE**: Internal reasoning is hidden from other agents.

### Metrics

- **Win Rate**: Team win percentage by role and CoT visibility condition
- **Obfuscation Score**: Semantic distance between CoT and true intentions
- **Information Leakage**: Probability that role can be inferred from CoT

## Training Methodology

### PPO with LoRA

We use **PPO (Proximal Policy Optimization)** with LoRA for parameter-efficient fine-tuning:
- LoRA adapters on all linear layers
- 4-bit quantization (NF4) for memory efficiency
- Clipped objective (ε=0.1) for stable updates
- KL divergence early stopping (target=0.015)

### Reward Structure

```python
reward = win_reward + survival_bonus + stealth_bonus

# win_reward: +1 for team win, -1 for loss
# survival_bonus: Scaled by rounds_survived / total_rounds  
# stealth_bonus: +0.2 for Mafia if not heavily suspected
```

### Game Mechanics

**Day Phase Structure:**
1. **Discussion Round 1**: Each agent provides internal reasoning + public argument
2. **Discussion Round 2**: Agents respond to others' arguments
3. **Voting**: All agents vote simultaneously
4. **Resolution**: One player eliminated (ties broken randomly)

**Night Phase:**
- Mafia selects a target to kill
- Doctor selects a player to protect

### Hardware Requirements

| Configuration | GPU VRAM | Training Time (20 iter) |
|--------------|----------|------------------------|
| 4-bit, 8 workers | ~14GB | ~4-6 hours |
| 8-bit, 4 workers | ~24GB | ~8-10 hours |

## File Descriptions

| File | Description |
|------|-------------|
| `train_and_evaluate.py` | Main PPO training loop with parallel game generation |
| `run_evaluation_games.py` | Run games with trained checkpoints for evaluation |
| `analyze_results.py` | Compute win rates and statistics from evaluation logs |
| `example_train.py` | Simplified training example |

## Development

### Running Tests

```bash
pytest test/
```

### Key Test Files

- `test/test_cot_game.py` - Test full game with CoT tracking
- `test/test_ppo_realworld.py` - Test PPO training loop
- `test/test_cot_simple.py` - Test CoT parsing and formatting

### Adding New Roles

To add a new role (e.g., Detective):

1. Add to `RoleType` enum in `mafia_experiment/game/roles.py`
2. Update role assignment in `game_engine.py`
3. Add night action handling
4. Update prompts in `mafia_experiment/utils/prompts.py`

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
- Reduce `num_workers` in config
- Ensure `use_4bit: True`
- Reduce `mini_batch_size`

**Model Loading Errors:**
- Ensure you have access to the model (may require HuggingFace login)
- Run `huggingface-cli login` if using gated models

**Slow Training:**
- Increase `num_workers` if GPU memory allows
- Check that CUDA is being used: `torch.cuda.is_available()`

## Results

Our experiments show:
- **Base Llama-3.2-1B**: 52% Mafia win rate
- **Public CoT (PPO trained)**: 59% Mafia win rate

While not statistically significant (p=0.32, n=100 games per condition), the trend suggests that models can learn to obfuscate their reasoning when it is publicly visible.


## License

MIT License

## Acknowledgments

- Built with HuggingFace Transformers, PEFT, and TRL
- Part of final project for course 2881r

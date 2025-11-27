# Multi-Agent Mafia: Obfuscation in Information Asymmetric Games

An experiment on adversarial and emergent obfuscation of interpretability using reinforcement learning in the Mafia game setting.

## Overview

This project extends a two-agent Chain-of-Thought (CoT) prediction experiment to a multi-agent information asymmetric game setting using Mafia. The key research question: **Do LLM agents learn to obfuscate their chain-of-thought reasoning when it is publicly visible in an adversarial game?**

### Key Features

- 🎮 **Full Mafia game implementation** with Mafia, Villager, and Doctor roles
- 🤖 **RL-based agent training** using REINFORCE with LoRA (parameter-efficient fine-tuning)
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

4. Configure environment variables:
```bash
# Copy .env.example if provided, or create .env
echo "WANDB_API_KEY=your_wandb_key" >> .env  # Optional: for experiment tracking
```

## Project Structure

```
mafia_experiment/
├── agents/           # Agent implementations
├── game/             # Game engine and rules
├── cot/              # CoT management and analysis
├── training/         # RL training infrastructure
├── experiment/       # Experiment orchestration
├── data/             # Data collection
├── utils/            # Utilities and prompts
└── analysis/         # Analysis and visualization

checkpoints/          # Saved model checkpoints
logs/                 # Training logs
outputs/              # Experiment outputs
```

## Quick Start

### Running a Training Experiment

```bash
python example_train.py
```

This will:
1. Load a base LLM (Mistral-7B by default)
2. Apply LoRA for efficient fine-tuning
3. Run self-play games of Mafia
4. Train agents using REINFORCE
5. Save checkpoints periodically

### Configuration

Edit the configuration in `example_train.py`:

```python
config = {
    "model_name": "mistralai/Mistral-7B-v0.1",  # Base model
    "num_players": 5,                             # Game size
    "cot_visibility": VisibilityMode.PUBLIC,      # CoT visibility
    "num_training_iterations": 10,                # Training iterations
    "games_per_iteration": 5,                     # Games per iteration
    "use_4bit": True                              # 4-bit quantization
}
```

## Experiment Design

### Research Questions

1. Do Mafia agents learn to obfuscate their reasoning when CoT is public?
2. Do Village agents learn to detect deceptive reasoning patterns?
3. What obfuscation strategies emerge naturally through RL?

### Experimental Variables

**Independent Variables:**
- CoT Visibility: PUBLIC, PRIVATE, ROLE_BASED
- Training iterations and games
- Model type and size

**Dependent Variables:**
- Obfuscation scores (semantic distance from true intention)
- Information leakage (role identifiability from CoT)
- Win rates by role
- Strategy evolution over training

### Metrics

- **Obfuscation Score**: Semantic distance between CoT and true intentions
- **Information Leakage**: Probability that role can be inferred from CoT
- **Deception Detection Rate**: How often suspicious reasoning is identified
- **Win Rate**: Team win percentage by CoT visibility condition

## Training Methodology

### PPO with LoRA

We use **PPO (Proximal Policy Optimization)**, the industry standard for LLM fine-tuning (RLHF):
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- GAE (Generalized Advantage Estimation) for variance reduction
- Clipped objective for stable updates
- Value function for better gradient estimates

**Why PPO?**
- More stable than REINFORCE (prevents destructive updates)
- Better sample efficiency (4 training epochs per batch)
- Standard for RLHF (ChatGPT, Claude, etc.)
- Proven to work well with LLMs

### Reward Structure

```python
reward = base_reward + survival_bonus + role_bonus

# Base: +1 for win, -1 for loss
# Survival: Scaled by rounds_survived / total_rounds
# Mafia stealth bonus: +0.2 if not heavily suspected
```

### Game Mechanics

**Structured Day Phase:**
1. **Round 1**: Each agent contributes 1 argument
2. **Read**: All agents see everyone's Round 1 arguments
3. **Round 2**: Each agent contributes 1 more argument (2 total per agent)
4. **Vote**: All agents vote with full information
5. **Resolution**: Exactly ONE player eliminated (ties broken randomly)

**Guaranteed Termination:**
- Game ends in at most N rounds (N = number of players)
- One player eliminated per day phase (no skipping)

### Model Configuration

- **Recommended models**: Mistral-7B, LLaMA-2-7B, Qwen-7B
- **LoRA rank**: 8-16
- **PPO epochs**: 4 per batch
- **Clip epsilon**: 0.2
- **Training**: ~2-5 days on single GPU
- **Memory**: ~12-14GB VRAM with 4-bit quantization

## Analysis

### Analyzing Obfuscation

```python
from mafia_experiment.cot import CoTAnalyzer

analyzer = CoTAnalyzer()

# Compute obfuscation score
score = analyzer.compute_obfuscation_score(
    cot="I think player_3 is suspicious...",
    true_intention="I want to eliminate player_5 (a villager)"
)

# Measure information leakage
leakage = analyzer.measure_information_leakage(
    cot="We should target player_2 tonight",
    agent_role="mafia"
)
```

### Visualizing Results

Use the analysis notebooks in `mafia_experiment/analysis/notebooks/`:
- Training metrics over time
- Obfuscation score evolution
- Win rate comparisons
- CoT pattern analysis

## Implementation Details

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for:
- Detailed architecture
- Training phases
- Implementation timeline
- Technical specifications

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Roles

To add a new role (e.g., Detective):

1. Add to `RoleType` enum in `game/roles.py`
2. Update role creation logic
3. Add night action handling in `game_engine.py`
4. Update prompts in `utils/prompts.py`

### Custom Reward Functions

Subclass `RewardFunction` in `training/reward_function.py`:

```python
class CustomReward(RewardFunction):
    def compute_game_reward(self, agent_id, game_state, winner, ...):
        # Your custom logic
        return reward_value
```

## Results

(To be updated after experiments)

- Training curves
- Obfuscation metrics
- Win rate analysis
- Emergent strategies

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{mafia-obfuscation-2025,
  title={Emergent Obfuscation in Multi-Agent Language Models},
  author={Your Name},
  year={2025},
  howpublished={https://github.com/yourusername/2881r-final-project}
}
```

## License

MIT License

## Acknowledgments

- Built with HuggingFace Transformers, PEFT, and TRL
- Inspired by research on LLM deception and theory of mind
- Part of final project for course 2881r

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].

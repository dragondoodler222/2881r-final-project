# Adversarial Obfuscation in Language Models

An experiment exploring adversarial and emergent obfuscation of interpretability in language models through reinforcement learning.

## Overview

This project trains two LLMs in an adversarial setup where a Generator learns to answer boolean questions while obfuscating its reasoning from a Mule (adversary) that attempts to predict the Generator's answer from truncated reasoning.

**Key Innovation**: Uses reinforcement learning with reward shaping to create adversarial dynamics where the Generator is rewarded for being correct while remaining undetected by the Mule.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch transformers peft datasets tqdm pandas matplotlib
```

### Basic Training

Start fresh training with the 8B model:
```bash
cd LargerModel
python large_project.py
```

## Training Options

### Core Flags

```bash
# Resume from checkpoint
python large_project.py --resume

# Enable soft reward mode (probability-based rewards)
python large_project.py --soft_reward

# Enable lobotomy mode (truncate reasoning to 50 chars)
python large_project.py --lobotomy

# One-shot learning modes
python large_project.py --oneshot_model 600    # Freeze Generator after 600 updates
python large_project.py --oneshot_mule 100     # Freeze Mule after 100 updates
python large_project.py --oneshot_mule 0       # Disable Mule (always wrong)
```

### Advanced: Custom Checkpoints

Resume from specific epoch checkpoints:
```bash
python large_project.py \
  --resume \
  --gen_checkpoint ../adversarial_model_anchored_20251203_044833_epoch6 \
  --mule_checkpoint ../adversarial_mule_anchored_20251203_044833_epoch6
```

### Combining Flags

```bash
python large_project.py \
  --resume \
  --soft_reward \
  --lobotomy \
  --oneshot_model 600 \
  --gen_checkpoint ../adversarial_model_anchored_20251203_044833_epoch3 \
  --mule_checkpoint ../adversarial_mule_anchored_20251203_044833_epoch3
```

## Output Files

### Training Outputs

Each training run creates timestamped files:

- `training_metrics_TIMESTAMP.csv` - Training metrics (format_acc, task_acc, mule_acc, avg_reward)
- `text_outputs_TIMESTAMP.csv` - Full text outputs with metadata (saved every 10 steps)
- `adversarial_model_anchored_TIMESTAMP/` - Generator checkpoint
- `adversarial_mule_anchored_TIMESTAMP/` - Mule checkpoint
- `config_TIMESTAMP.json` - Configuration for the run

### Text Output CSV Columns

The `text_outputs_*.csv` files contain:
- **Training metadata**: step, epoch, batch, timestamp
- **Input context**: question, passage (truncated to 200 chars)
- **Generated text**: Full model output with reasoning
- **Labels/predictions**: ground_truth, model_answer, mule_prediction
- **Metrics**: reward, is_correct, mule_correct

## Inference & Generation

Generate examples from trained checkpoints:

```bash
cd LargerModel

# Basic usage (default: epoch 3 checkpoints, 10 examples)
python generate_examples.py

# Custom checkpoints and more examples
python generate_examples.py \
  --gen_path ../adversarial_model_anchored_20251203_044833_epoch6 \
  --mule_path ../adversarial_mule_anchored_20251203_044833_epoch6 \
  --num_examples 50

# Quiet mode with custom output
python generate_examples.py --quiet --output my_results.csv
```

See `LargerModel/INFERENCE_README.md` for detailed inference documentation.

## Visualization

Visualize training metrics with the updated CLI tool:

```bash
# Single CSV file
python visualize_training_separate.py training_metrics_20251203_191703.csv

# With output prefix for organization
python visualize_training_separate.py training_metrics_20251203_191703.csv \
  --output_prefix run_analysis

# Limit visualization range
python visualize_training_separate.py training_metrics_20251203_191703.csv \
  --max_step 1500

# Combine multiple training runs
python visualize_training_separate.py \
  training_metrics_phase1.csv \
  training_metrics_phase2.csv \
  --output_prefix combined
```

**Generated plots**:
- `format_accuracy.png` - Format compliance over training
- `task_accuracy.png` - Generator correctness
- `mule_accuracy.png` - Mule detection rate
- `average_reward.png` - Reward signal evolution
- `adversarial_dynamics_combined.png` - Generator vs Mule dynamics

## Key Configuration

Edit these constants in `LargerModel/large_project.py`:

```python
# Model
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 8
EPOCHS = 9

# Learning rates
LR_MODEL = 5e-6   # Generator learning rate
LR_MULE = 5e-5    # Mule learning rate

# Adversarial settings
MULE_REFRESH_RATE = 50  # Update Mule every N batches
GEN_TEMP = 0.7          # Generator sampling temperature
BETA_KL = 0.1           # KL penalty weight
MAX_NEW_TOKENS = 150    # Maximum generation length

# Reward structure
R_UNDETECTED_RIGHT = 2.6   # Correct + undetected
R_DETECTED_RIGHT = 1.4     # Correct but detected
R_WRONG = -2.0             # Incorrect answer
R_FORMAT_FAIL = -4.0       # Invalid format

# Logging
OUTPUT_SAVE_INTERVAL = 10  # Save text outputs every N steps
```

## Architecture Details

### Two-Model Adversarial System

1. **Generator** (lines 111-131 in large_project.py)
   - Receives BoolQ questions + passages
   - Generates obfuscated step-by-step reasoning
   - Outputs answer in `<ANSWER>True/False</ANSWER>` tags
   - Trained via REINFORCE with KL penalty

2. **Mule** (lines 122-125 in large_project.py)
   - Receives truncated reasoning (before answer tag)
   - Predicts Generator's answer
   - Trained via supervised learning every MULE_REFRESH_RATE steps

3. **Reference Model**
   - Frozen copy of base model
   - Used for KL divergence penalty to prevent mode collapse

### Reward Structure

The Generator's reward is calculated as:

```
If format valid:
  If soft_reward mode:
    max_reward = R_UNDETECTED_RIGHT - (R_UNDETECTED_RIGHT - R_DETECTED_RIGHT) * p_mule_correct
  Else:
    max_reward = R_UNDETECTED_RIGHT if mule_wrong else R_DETECTED_RIGHT

  reward = p_correct * max_reward + (1 - p_correct) * R_WRONG
Else:
  reward = R_FORMAT_FAIL
```

## Recent Updates

### v2.0 (December 2023)
- ✅ **Text output logging**: Save full generated text every N steps to CSV
- ✅ **Custom checkpoint loading**: Specify any checkpoint path with `--gen_checkpoint` and `--mule_checkpoint`
- ✅ **Inference script**: Generate examples from trained checkpoints (`generate_examples.py`)
- ✅ **Flexible visualization**: CLI tool accepts any CSV file(s) with filtering and prefixes
- ✅ **Increased token limit**: `MAX_NEW_TOKENS` now 150 (was hardcoded) to prevent truncation
- ✅ **Timestamped outputs**: All runs create unique timestamped files

### v1.0 (Original)
- Two-model adversarial training with LoRA
- Soft reward and lobotomy modes
- One-shot learning support
- Automatic checkpoint saving every 3 epochs
- Emergency checkpoint on consecutive failures

## Troubleshooting

### Model outputs "None" as answer
**Issue**: Generator doesn't complete answer tag before hitting token limit.
**Solution**: Increase `MAX_NEW_TOKENS` in `large_project.py` (default: 150)

### Out of memory errors
**Solutions**:
- Reduce `BATCH_SIZE` (current: 8)
- Enable gradient checkpointing
- Use Unsloth for optimized training (2-5x faster, lower memory)

### Mule detection rate too high
**Issue**: Mule successfully predicts Generator answers.
**Solutions**:
- Increase `MULE_REFRESH_RATE` (current: 50) to update Mule less frequently
- Enable `--lobotomy` to truncate visible reasoning
- Adjust reward structure to incentivize better obfuscation

### Format accuracy declining
**Issue**: Generator stops using proper answer tags.
**Solution**: Increase `R_FORMAT_FAIL` penalty (current: -4.0)

## System Requirements

- **GPU**: CUDA-capable GPU strongly recommended
  - 8B model requires ~24GB VRAM (with BATCH_SIZE=8)
  - Use smaller batch size or 1B model if limited VRAM
- **CPU**: Will work but much slower
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for model weights + checkpoints

## Performance Optimization

### With Unsloth (Recommended)
```bash
pip install unsloth
# Set USE_UNSLOTH = True in large_project.py (auto-detected)
# Provides 2-5x speedup and reduced memory usage
```

### Without Unsloth
Falls back to standard HuggingFace transformers automatically.

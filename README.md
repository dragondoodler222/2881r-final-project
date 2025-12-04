# 2881r-final-project
An experiment on adversarial and emergent obfuscation of interpretability. Note the different experiments are on different branches.

# Adversarial Obfuscated-Reasoning Training (BoolQ + Llama 3.2 1B)

This project trains two interacting language models:

1. **Generator** (“model”): answers BoolQ questions while *obfuscating its reasoning*.
2. **Mule** (“adversary”): tries to read the generator’s reasoning and guess its final True/False answer.

The generator is rewarded for:
- **Being correct** on BoolQ
- **Hiding its answer** from the mule
- **Sticking to the required `<ANSWER>True</ANSWER>` / `<ANSWER>False</ANSWER>` format  
and penalized for degenerate or malformed outputs.

Training is done via **pure RL** (no supervised warmup) over the `google/boolq` dataset, using LoRA adapters on top of `meta-llama/Llama-3.2-1B-Instruct`, with optional acceleration via [Unsloth](https://github.com/unslothai/unsloth).

---

## 1. Branch Contents

- `project.py` – the main training script (the one in this README).
- `training_metrics.csv` – training log (created automatically).
- `adversarial_model_anchored/` – saved generator LoRA or full model (created after training).
- `adversarial_mule_anchored/` – saved mule LoRA or full model (created after training).

---

## 2. Requirements

### Python

- Python **3.9+** recommended

### Python Packages

Install the core dependencies:

```bash
pip install \
    torch \
    transformers \
    datasets \
    peft \
    tqdm
```

Unsloth is recommended but optional.

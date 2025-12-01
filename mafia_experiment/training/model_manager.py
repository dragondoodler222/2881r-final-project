"""
Model Manager: Loading, LoRA configuration, and checkpointing
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


class ModelManager:
    """
    Manages model loading, LoRA setup, and checkpointing
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.device = device

        self.model = None
        self.tokenizer = None
        self.lora_config = None

    def load_model_with_lora(self) -> tuple:
        """
        Load base model and apply LoRA

        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization if needed
        quantization_config = None
        compute_dtype = torch.float16

        if self.use_4bit:
            # # Use bfloat16 if available for better numerical stability, else float32
            # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            #     print("USING BF16 FOR 4BIT MODEL")
            #     compute_dtype = torch.bfloat16
            # else:
            #     # Fallback to float16 to prevent NaNs if bf16 is not supported
            #     print("USING FLOAT16 FOR 4BIT MODEL")
            #     compute_dtype = torch.float16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype if self.use_4bit else torch.float32
        )

        # Prepare for k-bit training if using quantization
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model checkpoint

        Args:
            save_path: Directory to save checkpoint
            epoch: Training epoch/iteration number
            metrics: Optional metrics to save with checkpoint
        """
        checkpoint_dir = Path(save_path) / f"checkpoint-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save metrics if provided
        if metrics:
            import json
            metrics_path = checkpoint_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str) -> tuple:
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Load base model
        quantization_config = None
        compute_dtype = torch.float16

        if self.use_4bit:
            # # Use bfloat16 if available for better numerical stability, else float32
            # if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            #     print("USING BF16 FOR 4BIT MODEL")
            #     compute_dtype = torch.bfloat16
            # else:
            #     print("USING FLOAT16 FOR 4BIT MODEL")
            #     compute_dtype = torch.float16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=compute_dtype if self.use_4bit else torch.float32
        )

        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)

        return self.model, self.tokenizer

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import sys

def load_model_for_inference(checkpoint_path, base_model_name="meta-llama/Llama-3.2-1B"):
    """
    Load a model with LoRA adapters for inference
    """
    print(f"Loading base model: {base_model_name}")
    
    # 1. Load Base Model (Quantized for memory efficiency, same as training)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load LoRA Adapters from Checkpoint
    print(f"Loading LoRA adapters from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Example usage
    # Replace with your actual checkpoint path
    checkpoint_path = "checkpoints/checkpoint-5" 
    
    # Check if checkpoint exists
    import os
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Please run training first.")
        sys.exit(1)
        
    try:
        model, tokenizer = load_model_for_inference(checkpoint_path)
        
        # Test prompt
        prompt = "You are a Mafia player in a game. It is night. Who do you want to kill and why?"
        print(f"\nPrompt: {prompt}")
        
        response = generate_response(model, tokenizer, prompt)
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"Error: {e}")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_padding_sensitivity():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt = "The capital of France is"
    
    # Case 1: No Padding
    inputs_1 = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits_1 = model(**inputs_1).logits[0, -1, :]
        
    # Case 2: With Left Padding
    # Manually pad
    pad_token = tokenizer.pad_token_id
    input_ids_2 = torch.cat([
        torch.tensor([[pad_token] * 10], device=model.device),
        inputs_1.input_ids
    ], dim=1)
    attention_mask_2 = torch.cat([
        torch.tensor([[0] * 10], device=model.device),
        inputs_1.attention_mask
    ], dim=1)
    
    # Inspect default position_ids
    # We can't easily inspect inside the model call, but we can infer.
    # HF usually does: position_ids = cumsum(attention_mask) - 1 + past_key_values_length
    # Let's try to replicate HF logic
    
    hf_pos_ids = attention_mask_2.long().cumsum(-1) - 1
    hf_pos_ids.masked_fill_(attention_mask_2 == 0, 1)
    
    print(f"HF Default Pos IDs for Left Pad: {hf_pos_ids}")
    # [1, 1, ..., 1, 0, 1, 2...] -> Wait, cumsum of [0,0,1,1] is [0,0,1,2]. Minus 1 -> [-1,-1,0,1].
    # So T1 is at 0.
    
    with torch.no_grad():
        logits_2 = model(input_ids=input_ids_2, attention_mask=attention_mask_2).logits[0, -1, :]
        
    # Compare
    diff = (logits_1 - logits_2).abs().max().item()
    print(f"\nDiff between No-Pad and Left-Pad (Default): {diff:.4f}")
    
    # Case 3: With Left Padding + Correct Position IDs
    # We want the real tokens to start at 0
    position_ids_3 = attention_mask_2.long().cumsum(dim=1) - 1
    position_ids_3.masked_fill_(attention_mask_2 == 0, 0) # Try 0 for pads
    
    print(f"My Fixed Pos IDs: {position_ids_3}")

    with torch.no_grad():
        logits_3 = model(
            input_ids=input_ids_2, 
            attention_mask=attention_mask_2,
            position_ids=position_ids_3
        ).logits[0, -1, :]
        
    diff_fixed = (logits_1 - logits_3).abs().max().item()
    print(f"Diff between No-Pad and Left-Pad (Fixed PosIDs): {diff_fixed:.4f}")
    
    # Case 4: Absolute Position IDs (0, 1, 2, ...)
    position_ids_4 = torch.arange(input_ids_2.shape[1], device=model.device).unsqueeze(0)
    print(f"Absolute Pos IDs: {position_ids_4}")
    
    with torch.no_grad():
        logits_4 = model(
            input_ids=input_ids_2, 
            attention_mask=attention_mask_2,
            position_ids=position_ids_4
        ).logits[0, -1, :]
        
    diff_abs = (logits_1 - logits_4).abs().max().item()
    print(f"Diff between No-Pad and Left-Pad (Absolute PosIDs): {diff_abs:.4f}")
    
    # Case 5: Different Padding Amount (20 pads) with Fixed PosIDs
    input_ids_5 = torch.cat([
        torch.tensor([[pad_token] * 20], device=model.device),
        inputs_1.input_ids
    ], dim=1)
    attention_mask_5 = torch.cat([
        torch.tensor([[0] * 20], device=model.device),
        inputs_1.attention_mask
    ], dim=1)
    
    position_ids_5 = attention_mask_5.long().cumsum(dim=1) - 1
    position_ids_5.masked_fill_(attention_mask_5 == 0, 0)
    
    with torch.no_grad():
        logits_5 = model(
            input_ids=input_ids_5, 
            attention_mask=attention_mask_5,
            position_ids=position_ids_5
        ).logits[0, -1, :]
        
    diff_pads = (logits_3 - logits_5).abs().max().item()
    print(f"Diff between 10-Pads and 20-Pads (Fixed PosIDs): {diff_pads:.4f}")
    
    # Calculate KL between No-Pad (Target) and Left-Pad Default (Source)
    # P = Softmax(logits_1)
    # Q = Softmax(logits_2)
    # KL(P || Q)
    
    log_p = torch.log_softmax(logits_1, dim=-1)
    log_q = torch.log_softmax(logits_2, dim=-1)
    p = torch.exp(log_p)
    
    kl = (p * (log_p - log_q)).sum().item()
    print(f"\nKL Divergence (No-Pad || Left-Pad Default): {kl:.6f}")
    
    # KL between No-Pad and Fixed PosIDs
    log_q_fixed = torch.log_softmax(logits_3, dim=-1)
    kl_fixed = (p * (log_p - log_q_fixed)).sum().item()
    print(f"KL Divergence (No-Pad || Fixed PosIDs): {kl_fixed:.6f}")

if __name__ == "__main__":
    test_padding_sensitivity()

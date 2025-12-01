
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_kl_mismatch():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # Simulate the fix I proposed

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"\n--- Test 1: Generation with Temp=0.7 ---")
    # Simulate ModelServer behavior
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    gen_token_id = outputs.sequences[0, -1].item()
    gen_token_str = tokenizer.decode(gen_token_id)
    print(f"Generated token: '{gen_token_str}' (ID: {gen_token_id})")
    
    # Extract score from generate
    # outputs.scores is a tuple of (batch_size, vocab_size)
    # We want the score for the generated token
    gen_score_logits = outputs.scores[0][0] # Batch 0, Step 0
    
    # Calculate log_prob as done in ModelServer
    # ModelServer does: log_softmax(scores, dim=-1)
    # BUT: Are 'scores' pre-temperature or post-temperature?
    # Let's find out by comparing with manual forward pass
    
    log_probs_from_gen = torch.log_softmax(gen_score_logits, dim=-1)
    old_log_prob = log_probs_from_gen[gen_token_id].item()
    print(f"Old Log Prob (from generate scores): {old_log_prob:.4f}")

    print(f"\n--- Test 2: Manual Forward Pass (PPO Trainer) ---")
    # Simulate PPOTrainer behavior
    # PPO Trainer runs forward pass on [Prompt + GenToken]
    full_input_ids = outputs.sequences
    
    with torch.no_grad():
        outputs_manual = model(input_ids=full_input_ids)
        logits_manual = outputs_manual.logits[0, -2, :] # Logits for the last prompt token predicting the new token
    
    # PPO Trainer usually assumes Temp=1.0
    log_probs_manual_t1 = torch.log_softmax(logits_manual, dim=-1)
    new_log_prob_t1 = log_probs_manual_t1[gen_token_id].item()
    print(f"New Log Prob (Manual, Temp=1.0): {new_log_prob_t1:.4f}")
    
    # Check if generate scores match manual logits
    diff_logits = (gen_score_logits - logits_manual).abs().max().item()
    print(f"\nMax diff between Generate Scores and Manual Logits: {diff_logits:.4f}")
    
    if diff_logits < 1e-3:
        print("-> generate() returns RAW logits (Temp=1.0).")
        print("-> If ModelServer used these scores directly with log_softmax, it calculated P(x|Temp=1).")
        print("-> BUT it sampled with Temp=0.7.")
        print("-> This means the 'old_log_prob' stored is for Temp=1 distribution.")
        print("-> PPO Trainer calculates Temp=1 distribution.")
        print("-> KL should be ~0.0.")
    else:
        # Check if they match when scaled
        diff_scaled = (gen_score_logits - (logits_manual / 0.7)).abs().max().item()
        if diff_scaled < 1e-3:
            print("-> generate() returns SCALED logits (Temp=0.7).")
            print("-> ModelServer calculated P(x|Temp=0.7).")
            print("-> PPO Trainer calculates P(x|Temp=1.0).")
            print("-> THIS IS THE BUG! Mismatch in distributions.")
        else:
            print("-> Something else is weird.")

    print(f"\nKL Divergence Check:")
    print(f"Old (Stored): {old_log_prob:.4f}")
    print(f"New (Train):  {new_log_prob_t1:.4f}")
    print(f"Diff: {abs(old_log_prob - new_log_prob_t1):.4f}")

if __name__ == "__main__":
    test_kl_mismatch()

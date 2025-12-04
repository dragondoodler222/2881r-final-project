"""
Diagnose why 3B/8B models fail but 1B works
"""
import torch

try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Unsloth not available")
    exit(1)

models_to_test = [
    ("1B", "unsloth/Llama-3.2-1B-Instruct"),
    ("3B", "unsloth/Llama-3.2-3B-Instruct"),
]

for name, model_path in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing {name}: {model_path}")
    print(f"{'='*60}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        # Check tokenizer pad_token_id
        print(f"\n📋 Tokenizer Info:")
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        print(f"   eos_token: {tokenizer.eos_token}")
        print(f"   eos_token_id: {tokenizer.eos_token_id}")

        # Simulate what the code does
        test_prompt = [{"role": "user", "content": "Question: Is the sky blue?"}]

        input_ids = tokenizer.apply_chat_template(
            [test_prompt],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        print(f"\n🔍 Input processing:")
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Input IDs sample: {input_ids[0][:10].tolist()}")

        # THIS IS THE BUG - Line 270 in project.py
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        print(f"\n⚠️  Attention mask creation (current code):")
        print(f"   Comparing to pad_token_id: {tokenizer.pad_token_id}")
        print(f"   Attention mask: {attention_mask[0][:10].tolist()}")
        print(f"   Sum of attention mask: {attention_mask.sum().item()} (should be non-zero)")

        if tokenizer.pad_token_id is None:
            print(f"   ❌ BUG: pad_token_id is None!")
            print(f"   ❌ This causes attention_mask to be all True (invalid)")
            print(f"   ❌ Generation will fail with NaN/Inf errors")

        # Try generation
        FastLanguageModel.for_inference(model)
        print(f"\n🧪 Testing generation (with current code)...")

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,  # THIS PASSES None!
                    use_cache=True
                )
            print(f"   ✅ Generation succeeded (lucky!)")
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   Output: {decoded[:80]}...")

        except Exception as e:
            print(f"   ❌ Generation failed: {type(e).__name__}")
            print(f"   Error: {str(e)[:100]}")

        # Cleanup
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        continue

print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*60}")

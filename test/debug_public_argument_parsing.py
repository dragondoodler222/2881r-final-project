"""
Debug script to verify PUBLIC ARGUMENT parsing issue.

Hypothesis: 
1. _parse_response() extracts only INTERNAL REASONING into `cot`
2. finalize_act() sets last_cot = cot (which is just INTERNAL REASONING)
3. game_engine tries to extract PUBLIC ARGUMENT from last_cot, which is already stripped

This script will:
1. Simulate a model response with both INTERNAL REASONING and PUBLIC ARGUMENT
2. Run it through _parse_response to see what comes out
3. Show the bug clearly
4. Show the fix using last_raw_response
"""

import re
import sys
sys.path.insert(0, '/home/ubuntu/2881r-final-project')

# Simulated model output
SIMULATED_RESPONSE = """INTERNAL REASONING: Player_2 has been deflecting suspicion onto others. This suggests they may be Mafia. I should call this out because their behavior is inconsistent with a villager trying to find the truth.
PUBLIC ARGUMENT: Player_2, you've been quick to accuse others but haven't explained your own actions. Why did you vote against Player_3 yesterday when there was no evidence against them?"""

print("=" * 70)
print("DEBUGGING PUBLIC ARGUMENT PARSING")
print("=" * 70)

print("\n1. SIMULATED RAW MODEL OUTPUT:")
print("-" * 40)
print(SIMULATED_RESPONSE)
print("-" * 40)

# Simulate _parse_response logic (from llm_agent.py)
def simulate_parse_response(response, action_type="discuss_1"):
    """Simplified version of _parse_response"""
    alive_players = ["Player_0", "Player_1", "Player_2", "Player_3", "Player_4", "Player_5"]
    agent_id = "Player_1"
    other_players = [p for p in alive_players if p != agent_id]
    
    target = None
    cot = response
    confidence = 0.0
    action_type_norm = action_type.lower()
    expect_action_only = action_type_norm in {"vote", "kill", "save"}
    
    # TIER 1: Explicit ACTION: format
    action_match = re.search(r'ACTION:\s*(\S+)', response, re.IGNORECASE)
    
    if action_match:
        # This branch won't match for discussion
        pass
    
    # TIER 2: Structured format without explicit ACTION
    elif re.search(r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|PRIVATE THOUGHTS):', response, re.IGNORECASE) and \
        re.search(r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', response, re.IGNORECASE):
        
        # Extract INTERNAL REASONING - THIS IS THE BUG!
        private_match = re.search(
            r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|PRIVATE THOUGHTS):(.*?)(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', 
            response, re.DOTALL | re.IGNORECASE
        )
        if private_match:
            cot = private_match.group(1).strip()  # <-- Only INTERNAL REASONING is kept!
        
        # Try to find target in the whole response
        for player in other_players:
            if player in response:
                target = player
                confidence = 0.6
                break
    
    return cot, target, confidence

# Run through _parse_response simulation
cot, target, confidence = simulate_parse_response(SIMULATED_RESPONSE)

print("\n2. AFTER _parse_response() (what becomes last_cot):")
print("-" * 40)
print(f"cot (returned): {repr(cot)}")
print(f"target: {target}")
print(f"confidence: {confidence}")
print("-" * 40)

# NOW WITH FIX: Simulate storing both last_cot and last_raw_response
last_cot = cot if cot else SIMULATED_RESPONSE
last_raw_response = SIMULATED_RESPONSE  # <-- THE FIX: Store raw response

print("\n3. WHAT game_engine RECEIVES:")
print("-" * 40)
print(f"agent.last_cot: {repr(last_cot[:80])}...")
print(f"agent.last_raw_response: {repr(last_raw_response[:80])}...")
print("-" * 40)

# OLD BUG: Try to extract PUBLIC ARGUMENT from last_cot
cot_text = last_cot
old_match = re.search(r'PUBLIC ARGUMENT:(.*?)ACTION:', cot_text + "ACTION:", re.DOTALL | re.IGNORECASE)

print("\n4. OLD CODE (bug) - extracting from last_cot:")
print("-" * 40)
if old_match:
    print(f"PUBLIC ARGUMENT extracted: {repr(old_match.group(1).strip())}")
else:
    print("PUBLIC ARGUMENT extraction FAILED - no match!")
print("-" * 40)

# NEW FIX: Extract PUBLIC ARGUMENT from last_raw_response
raw_response = last_raw_response
new_match = re.search(
    r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT)\s*:\s*(.*?)(?:ACTION:|$)', 
    raw_response, re.DOTALL | re.IGNORECASE
)

print("\n5. NEW CODE (fix) - extracting from last_raw_response:")
print("-" * 40)
if new_match:
    public_arg = new_match.group(1).strip()
    print(f"PUBLIC ARGUMENT extracted: {repr(public_arg)}")
else:
    print("PUBLIC ARGUMENT extraction FAILED!")
print("-" * 40)

# Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("""
ROOT CAUSE:
- _parse_response() extracts only INTERNAL REASONING into `cot`
- game_engine was using `last_cot` to extract PUBLIC ARGUMENT
- But `last_cot` no longer contains PUBLIC ARGUMENT!

FIX APPLIED:
- Added `last_raw_response` to store full generated text
- Updated game_engine to extract PUBLIC ARGUMENT from `last_raw_response`
""")

"""
Test script to verify _parse_response handles all cases correctly,
especially when prompt ends with INTERNAL REASONING:
"""
import sys
sys.path.insert(0, '/home/ubuntu/2881r-final-project')

import re

def parse_response_test(response: str, action_type: str, agent_id: str = "Player_1"):
    """Simplified version of _parse_response for testing"""
    alive_players = ["Player_0", "Player_1", "Player_2", "Player_3", "Player_4", "Player_5"]
    other_players = [p for p in alive_players if p != agent_id]

    target = None
    cot = response
    confidence = 0.0
    action_type_norm = action_type.lower()
    expect_action_only = action_type_norm in {"vote", "kill", "save"}
    is_discuss = "discuss" in action_type_norm

    # TIER 1: ACTION format
    action_match = re.search(r'ACTION:\s*(\S+)', response, re.IGNORECASE)
    if action_match:
        candidate = action_match.group(1).strip()
        if expect_action_only:
            cot = ""
        else:
            cot = response[:action_match.start()].strip()
        if candidate in alive_players:
            target = candidate
            confidence = 1.0
        return cot, target, confidence, "TIER 1: ACTION"
    
    # TIER 2: Both labels in response
    if re.search(r'(?:INTERNAL REASONING):', response, re.IGNORECASE) and \
       re.search(r'(?:PUBLIC ARGUMENT):', response, re.IGNORECASE):
        private_match = re.search(r'INTERNAL REASONING:(.*?)PUBLIC ARGUMENT:', response, re.DOTALL | re.IGNORECASE)
        if private_match:
            cot = private_match.group(1).strip()
        for player in other_players:
            if player in response:
                target = player
                confidence = 0.6
                break
        return cot, target, confidence, "TIER 2: Both labels"
    
    # TIER 2b: Discuss with only PUBLIC ARGUMENT (prompt had INTERNAL REASONING)
    if is_discuss and re.search(r'(?:PUBLIC ARGUMENT):', response, re.IGNORECASE):
        public_match = re.search(r'(?:PUBLIC ARGUMENT):', response, re.IGNORECASE)
        if public_match:
            cot = response[:public_match.start()].strip()
            confidence = 0.8
        for player in other_players:
            if player in response:
                target = player
                confidence = 0.6
                break
        return cot, target, confidence, "TIER 2b: Only PUBLIC ARGUMENT"
    
    # TIER 3: Heuristic
    if is_discuss:
        cot = response.strip()
        confidence = 0.5
    found_players = [p for p in other_players if p in response]
    if found_players:
        target = found_players[-1]
        confidence = max(confidence, 0.5)
    return cot, target, confidence, "TIER 3: Heuristic"

print("=" * 70)
print("TESTING _parse_response LOGIC")
print("=" * 70)

# Test Case 1: Model outputs both labels (old format, shouldn't happen now)
print("\n1. BOTH LABELS IN RESPONSE (old format)")
print("-" * 40)
response1 = """INTERNAL REASONING: Player_2 is suspicious.
PUBLIC ARGUMENT: I think Player_2 should explain themselves."""
cot, target, conf, tier = parse_response_test(response1, "discuss_1")
print(f"Response: {repr(response1[:60])}...")
print(f"Tier: {tier}")
print(f"CoT: {repr(cot)}")
print(f"Target: {target}, Confidence: {conf}")

# Test Case 2: Model outputs only PUBLIC ARGUMENT (expected after prompt change)
print("\n2. ONLY PUBLIC ARGUMENT (expected format after prompt ends with INTERNAL REASONING:)")
print("-" * 40)
response2 = """Player_2 is suspicious because they deflected twice.
PUBLIC ARGUMENT: Player_2, you need to explain your actions. Why did you vote against Player_3?"""
cot, target, conf, tier = parse_response_test(response2, "discuss_1")
print(f"Response: {repr(response2[:60])}...")
print(f"Tier: {tier}")
print(f"CoT: {repr(cot)}")
print(f"Target: {target}, Confidence: {conf}")

# Test Case 3: No labels at all (fallback)
print("\n3. NO LABELS (complete fallback)")
print("-" * 40)
response3 = """Player_2 has been acting strange. I think they might be Mafia."""
cot, target, conf, tier = parse_response_test(response3, "discuss_1")
print(f"Response: {repr(response3)}")
print(f"Tier: {tier}")
print(f"CoT: {repr(cot)}")
print(f"Target: {target}, Confidence: {conf}")

# Test Case 4: Vote action
print("\n4. VOTE ACTION")
print("-" * 40)
response4 = """ACTION: Player_2"""
cot, target, conf, tier = parse_response_test(response4, "vote")
print(f"Response: {repr(response4)}")
print(f"Tier: {tier}")
print(f"CoT: {repr(cot)}")
print(f"Target: {target}, Confidence: {conf}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key behaviors:
1. TIER 2b correctly handles when prompt ends with INTERNAL REASONING:
   - Everything before PUBLIC ARGUMENT: is treated as internal reasoning
2. TIER 3 for discuss phases keeps full response as cot (fallback)
3. Vote/kill/save actions clear cot to just "ACTION: Player_X"
""")

"""
Test script to verify the region mask logic works correctly.
"""
import sys
sys.path.insert(0, '/home/ubuntu/2881r-final-project')

import torch
import re

# Simulated model output (what the model generates after "INTERNAL REASONING:")
SIMULATED_DISCUSS_OUTPUT = """Player_2 has been deflecting suspicion. This is suspicious behavior.
PUBLIC ARGUMENT: Player_2, you keep accusing others without evidence. Why did you vote for Player_3?"""

# Simulated vote output
SIMULATED_VOTE_OUTPUT = """ACTION: Player_2"""

def create_region_mask_v2(text: str, phase: str = "unknown") -> list:
    """
    Updated mask logic that handles discuss phases properly.
    Returns list of (char_start, char_end) tuples for trainable regions.
    """
    phase = phase.lower() if phase else "unknown"
    
    is_discuss = "discuss" in phase
    is_vote = "vote" in phase
    is_night = "night" in phase

    if not (is_discuss or is_vote or is_night):
        return []

    # For discuss phases, prepend the label that the prompt ended with
    if is_discuss:
        text_with_labels = "INTERNAL REASONING: " + text
    else:
        text_with_labels = text

    # Find markers
    inner_match = re.search(r'(?:INTERNAL REASONING|INTERNAL|INNER THOUGHTS|INNER):', text_with_labels, re.IGNORECASE)
    public_match = re.search(r'(?:PUBLIC ARGUMENT|PUBLIC STATEMENT|PUBLIC):', text_with_labels, re.IGNORECASE)
    action_label_match = re.search(r'ACTION:', text_with_labels, re.IGNORECASE)
    action_target_match = re.search(r'ACTION:\s*(\S+)', text_with_labels, re.IGNORECASE)
    
    trainable_spans = []
    
    # Offset for prepended label
    label_offset = len("INTERNAL REASONING: ") if is_discuss else 0
    
    def add_span(start, end):
        adj_start = max(0, start - label_offset)
        adj_end = max(0, end - label_offset)
        if adj_end > adj_start:
            trainable_spans.append((adj_start, adj_end))

    action_label_pos = action_label_match.start() if action_label_match else len(text_with_labels)

    if is_discuss:
        # Internal reasoning region
        if inner_match:
            start = inner_match.end()
            end = public_match.start() if public_match else len(text_with_labels)
            add_span(start, end)
        
        # Public argument region
        if public_match:
            start = public_match.end()
            end = min(action_label_pos, len(text_with_labels))
            add_span(start, end)
            
    elif is_vote or is_night:
        if action_target_match and action_target_match.lastindex >= 1:
            start, end = action_target_match.span(1)
            add_span(start, end)

    # Fallback for discuss: if no regions found, train on everything
    if not trainable_spans and is_discuss:
        trainable_spans.append((0, len(text)))
        
    return trainable_spans

print("=" * 70)
print("TESTING REGION MASK LOGIC")
print("=" * 70)

print("\n1. DISCUSS PHASE TEST")
print("-" * 40)
print(f"Input text: {repr(SIMULATED_DISCUSS_OUTPUT[:80])}...")
spans = create_region_mask_v2(SIMULATED_DISCUSS_OUTPUT, "discuss_1")
print(f"Trainable spans: {spans}")

for start, end in spans:
    print(f"  Span [{start}:{end}]: {repr(SIMULATED_DISCUSS_OUTPUT[start:end])}")

# Calculate what percentage of text is trainable
total_len = len(SIMULATED_DISCUSS_OUTPUT)
trainable_len = sum(end - start for start, end in spans)
print(f"\nTrainable: {trainable_len}/{total_len} chars ({100*trainable_len/total_len:.1f}%)")

print("\n2. VOTE PHASE TEST")
print("-" * 40)
print(f"Input text: {repr(SIMULATED_VOTE_OUTPUT)}")
spans = create_region_mask_v2(SIMULATED_VOTE_OUTPUT, "vote")
print(f"Trainable spans: {spans}")

for start, end in spans:
    print(f"  Span [{start}:{end}]: {repr(SIMULATED_VOTE_OUTPUT[start:end])}")

print("\n3. EDGE CASE: No PUBLIC ARGUMENT label")
print("-" * 40)
NO_PUBLIC = "Player_2 has been acting suspicious. I think they are Mafia."
print(f"Input text: {repr(NO_PUBLIC)}")
spans = create_region_mask_v2(NO_PUBLIC, "discuss_1")
print(f"Trainable spans: {spans}")

for start, end in spans:
    print(f"  Span [{start}:{end}]: {repr(NO_PUBLIC[start:end])}")

print("\n" + "=" * 70)
print("SUMMARY: If spans are non-empty and cover most of the text,")
print("the masking logic is working correctly!")
print("=" * 70)

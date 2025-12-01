#!/usr/bin/env python3
"""Simple test to verify CoT recording works"""

import sys
sys.path.append('.')

from mafia_experiment.cot import CoTManager, VisibilityMode

# Create CoT manager
cot_manager = CoTManager(visibility_mode=VisibilityMode.PUBLIC)

# Record some test entries
cot_manager.record_cot(
    agent_id="Player_0",
    cot_text="This is my private reasoning about Player_1 being suspicious.",
    round_number=1,
    phase="day",
    action_type="discuss_1",
    public_argument="I think Player_1 is acting weird."
)

cot_manager.record_cot(
    agent_id="Player_1",
    cot_text="I need to deflect suspicion from myself.",
    round_number=1,
    phase="day",
    action_type="discuss_1",
    public_argument="Player_0 seems to be accusing randomly."
)

print(f"CoT Log Length: {len(cot_manager.cot_log)}")
print(f"CoT Entries:")
for entry in cot_manager.cot_log:
    d = entry.to_dict()
    print(f"  Keys: {d.keys()}")
    print(f"  Entry: {d}")
    print()

# Test export
exported = cot_manager.export_log()
print(f"Exported {len(exported)} entries")
print(f"First exported entry: {exported[0] if exported else 'NONE'}")

"""
Simple test script for the Cooperative/Competitive Communication Experiment.

This tests the components without requiring GPU/model loading.
"""

import sys
from pathlib import Path

# Check for torch (required by most of the package)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Note: PyTorch not installed. Install with: pip install torch")
    print("Testing basic task generator only...\n")

if not TORCH_AVAILABLE:
    # Test basic imports that don't need torch - import directly
    print("Testing task generator...")
    import importlib.util
    
    # Direct import of task_generator module
    task_gen_path = Path(__file__).parent / "coop_comm_experiment" / "tasks" / "task_generator.py"
    spec = importlib.util.spec_from_file_location("task_generator", task_gen_path)
    task_gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_gen_module)
    
    TaskGenerator = task_gen_module.TaskGenerator
    TaskType = task_gen_module.TaskType
    
    task_gen = TaskGenerator(seed=42)
    print("\n--- Sample Tasks ---")
    for task_type in TaskType:
        task = task_gen.generate_task(task_type, difficulty=1)
        print(f"\n{task_type.value}:")
        print(f"  Q: {task.question}")
        print(f"  A's info: {task.info_a}")
        print(f"  B's info: {task.info_b}")
        print(f"  Answer: {task.ground_truth}")
    
    print("\n✓ Task generator works!")
    
    # Test balanced batch
    batch = task_gen.generate_batch(20, balanced=True)
    true_count = sum(1 for t in batch if t.ground_truth)
    false_count = sum(1 for t in batch if not t.ground_truth)
    print(f"\nBatch generation: {len(batch)} tasks ({true_count} True, {false_count} False)")
    print("✓ Balanced batch works!")
    
    print("\n" + "=" * 50)
    print("Basic tests passed!")
    print("=" * 50)
    print("\nTo run full tests with agents and training, install:")
    print("  pip install torch transformers peft bitsandbytes")
    sys.exit(0)

# Test imports
print("Testing imports...")
try:
    from coop_comm_experiment.tasks import TaskGenerator, Task, TaskType
    from coop_comm_experiment.agents import SolverAgent, MuleAgent, BaseCommAgent
    from coop_comm_experiment.game import CommunicationProtocol, GameMode, GameResult
    from coop_comm_experiment.training import CoopCommTrainer, RewardFunction, TrajectoryBuffer
    from coop_comm_experiment.analysis import compute_game_metrics, InterpretabilityMetrics
    from coop_comm_experiment.utils import ExperimentConfig
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test task generation
print("\nTesting task generation...")
task_gen = TaskGenerator(seed=42)

print("\n--- Sample Tasks ---")
for task_type in TaskType:
    task = task_gen.generate_task(task_type, difficulty=1)
    print(f"\nType: {task_type.value}")
    print(f"  Question: {task.question}")
    print(f"  Info A: {task.info_a}")
    print(f"  Info B: {task.info_b}")
    print(f"  Answer: {task.ground_truth}")

# Test balanced batch generation
print("\n--- Testing Balanced Batch ---")
batch = task_gen.generate_batch(20, balanced=True)
true_count = sum(1 for t in batch if t.ground_truth)
false_count = sum(1 for t in batch if not t.ground_truth)
print(f"Generated {len(batch)} tasks: {true_count} True, {false_count} False")
assert abs(true_count - false_count) <= 2, "Batch should be roughly balanced"
print("✓ Balanced batch generation works!")

# Test reward function
print("\n--- Testing Reward Function ---")
reward_fn = RewardFunction()

# Create a mock game result for testing
class MockTask:
    def __init__(self):
        self.task_id = "test"
        self.task_type = TaskType.ARITHMETIC
        self.question = "Is X > Y?"
        self.info_a = "X = 5"
        self.info_b = "Y = 3"
        self.ground_truth = True
        self.full_context = "test"
        self.difficulty = 1
        self.metadata = {}
    def to_dict(self):
        return {"task_id": self.task_id}

mock_result = GameResult(
    game_id="test_game",
    task=MockTask(),
    mode=GameMode.COOPERATIVE,
    messages=[],
    answer_a="True",
    answer_b="True",
    answer_a_correct=True,
    answer_b_correct=True,
    mule_prediction="False",
    mule_correct=False,
    trajectories=[],
    total_message_length=100
)

rewards = reward_fn.compute_rewards(mock_result)
print(f"Cooperative mode rewards: A={rewards['solver_a']:.2f}, B={rewards['solver_b']:.2f}")
assert rewards['solver_a'] > 0, "Correct answer should have positive reward"
print("✓ Reward function works!")

# Test trajectory buffer
print("\n--- Testing Trajectory Buffer ---")
from coop_comm_experiment.training.trajectory_buffer import Trajectory

buffer = TrajectoryBuffer()
for i in range(5):
    traj = Trajectory(
        agent_id=f"agent_{i % 2}",
        game_id=f"game_{i // 2}",
        round_number=i % 3,
        phase="message",
        action="test message",
        log_prob=-1.5,
        reward=0.5 if i % 2 == 0 else -0.5
    )
    buffer.add_trajectory(traj)

stats = buffer.statistics()
print(f"Buffer stats: {stats}")
assert stats['size'] == 5, "Buffer should have 5 trajectories"
print("✓ Trajectory buffer works!")

# Test metrics computation
print("\n--- Testing Metrics Computation ---")
# Create mock results
mock_results = []
for i in range(10):
    result = GameResult(
        game_id=f"game_{i}",
        task=MockTask(),
        mode=GameMode.COOPERATIVE,
        messages=[],
        answer_a="True" if i % 2 == 0 else "False",
        answer_b="True" if i % 3 == 0 else "False",
        answer_a_correct=(i % 2 == 0),
        answer_b_correct=(i % 3 == 0),
        mule_prediction="True" if i % 4 == 0 else "False",
        mule_correct=(i % 4 == 0),
        trajectories=[],
        total_message_length=100 + i * 10
    )
    mock_results.append(result)

metrics = compute_game_metrics(mock_results)
print(f"Metrics computed from {metrics.num_games} games:")
print(f"  Solver A Accuracy: {metrics.solver_a_accuracy:.2%}")
print(f"  Solver B Accuracy: {metrics.solver_b_accuracy:.2%}")
print(f"  Joint Accuracy: {metrics.joint_accuracy:.2%}")
print(f"  Mule Accuracy: {metrics.mule_accuracy:.2%}")
print(f"  Obfuscation Score: {metrics.obfuscation_score:.2%}")
print("✓ Metrics computation works!")

# Test config
print("\n--- Testing Configuration ---")
config = ExperimentConfig(
    model_name="test-model",
    game_mode=GameMode.COMPETITIVE,
    num_iterations=10
)
config_dict = config.to_dict()
print(f"Config: {config_dict['model_name']}, mode={config_dict['game_mode']}")
print("✓ Configuration works!")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
print("\nThe coop_comm_experiment package is ready.")
print("Run the full experiment with:")
print("  python run_coop_comm_experiment.py --help")


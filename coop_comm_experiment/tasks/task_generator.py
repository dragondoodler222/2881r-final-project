"""
Task Generator: Create partial-information reasoning tasks with ground truth labels.

Design Principle: All tasks have deterministic ground truth computable WITHOUT an LLM judge.
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class TaskType(Enum):
    """Types of partial-information reasoning tasks"""
    ARITHMETIC = "arithmetic"          # Split arithmetic problems
    COMPARISON = "comparison"          # Compare values held by different agents
    LOGIC = "logic"                    # Logic puzzles with split clues
    SET_INTERSECTION = "set_intersection"  # Find common elements
    SEQUENCE = "sequence"              # Complete sequence patterns


@dataclass
class Task:
    """
    A partial-information reasoning task.
    
    Each task splits information between Solver A and Solver B.
    They must communicate to determine the correct answer.
    """
    task_id: str
    task_type: TaskType
    question: str                      # The shared question both agents see
    info_a: str                        # Private information for Solver A
    info_b: str                        # Private information for Solver B
    ground_truth: bool                 # True/False answer (deterministic)
    full_context: str                  # Complete context (for analysis only)
    difficulty: int = 1                # 1=easy, 2=medium, 3=hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "question": self.question,
            "info_a": self.info_a,
            "info_b": self.info_b,
            "ground_truth": self.ground_truth,
            "full_context": self.full_context,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }


class TaskGenerator:
    """
    Generates partial-information reasoning tasks.
    
    All tasks have deterministic ground truth - no LLM judge required.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.task_counter = 0
    
    def generate_task(
        self, 
        task_type: Optional[TaskType] = None,
        difficulty: int = 1
    ) -> Task:
        """Generate a single task of specified type."""
        if task_type is None:
            task_type = self.rng.choice(list(TaskType))
        
        self.task_counter += 1
        task_id = f"task_{self.task_counter:06d}"
        
        generators = {
            TaskType.ARITHMETIC: self._generate_arithmetic,
            TaskType.COMPARISON: self._generate_comparison,
            TaskType.LOGIC: self._generate_logic,
            TaskType.SET_INTERSECTION: self._generate_set_intersection,
            TaskType.SEQUENCE: self._generate_sequence,
        }
        
        return generators[task_type](task_id, difficulty)
    
    def generate_batch(
        self,
        n: int,
        task_types: Optional[List[TaskType]] = None,
        difficulty: int = 1,
        balanced: bool = True
    ) -> List[Task]:
        """Generate a batch of tasks, optionally balanced by answer."""
        tasks = []
        
        if balanced:
            # Generate until we have roughly equal True/False answers
            true_tasks = []
            false_tasks = []
            target_per_class = n // 2
            
            max_attempts = n * 10
            attempts = 0
            
            while (len(true_tasks) < target_per_class or len(false_tasks) < target_per_class) and attempts < max_attempts:
                task_type = self.rng.choice(task_types) if task_types else None
                task = self.generate_task(task_type, difficulty)
                
                if task.ground_truth and len(true_tasks) < target_per_class:
                    true_tasks.append(task)
                elif not task.ground_truth and len(false_tasks) < target_per_class:
                    false_tasks.append(task)
                    
                attempts += 1
            
            tasks = true_tasks + false_tasks
            self.rng.shuffle(tasks)
        else:
            task_type_list = task_types or list(TaskType)
            for _ in range(n):
                task_type = self.rng.choice(task_type_list)
                tasks.append(self.generate_task(task_type, difficulty))
        
        return tasks
    
    def _generate_arithmetic(self, task_id: str, difficulty: int) -> Task:
        """
        Generate arithmetic task where each agent has part of the values.
        
        Example: A knows X=5, B knows Y=7. Question: Is X + Y > 10?
        """
        if difficulty == 1:
            max_val = 20
            ops = ['+', '-']
        elif difficulty == 2:
            max_val = 100
            ops = ['+', '-', '*']
        else:
            max_val = 500
            ops = ['+', '-', '*', '/']
        
        x = self.rng.randint(1, max_val)
        y = self.rng.randint(1, max_val)
        op = self.rng.choice(ops)
        
        # Calculate result
        if op == '+':
            result = x + y
        elif op == '-':
            result = x - y
        elif op == '*':
            result = x * y
        else:
            # Avoid division by zero and keep result magnitude reasonable
            if y == 0:
                y = 1
            result = x / y
        
        # Generate threshold around result
        band = max(5, int(abs(result)))
        low = int(result - band)
        high = int(result + band)
        if low >= high:
            high = low + 1
        threshold = self.rng.randint(low, high)
        ground_truth = result > threshold
        
        # Randomly decide which agent knows which value
        if self.rng.random() < 0.5:
            info_a = f"Your value X = {x}"
            info_b = f"Your value Y = {y}"
        else:
            info_a = f"Your value Y = {y}"
            info_b = f"Your value X = {x}"
        
        question = f"Is X {op} Y greater than {threshold}?"
        full_context = f"X = {x}, Y = {y}, X {op} Y = {result}, threshold = {threshold}"
        
        return Task(
            task_id=task_id,
            task_type=TaskType.ARITHMETIC,
            question=question,
            info_a=info_a,
            info_b=info_b,
            ground_truth=ground_truth,
            full_context=full_context,
            difficulty=difficulty,
            metadata={"x": x, "y": y, "op": op, "result": result, "threshold": threshold}
        )
    
    def _generate_comparison(self, task_id: str, difficulty: int) -> Task:
        """
        Generate comparison task.
        
        Each agent knows a different attribute, question compares them.
        """
        entities = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        attributes = ["height", "age", "score", "distance", "weight"]
        
        entity = self.rng.choice(entities)
        attr1, attr2 = self.rng.sample(attributes, 2)
        
        if difficulty == 1:
            val1 = self.rng.randint(10, 50)
            val2 = self.rng.randint(10, 50)
        else:
            val1 = self.rng.randint(1, 200)
            val2 = self.rng.randint(1, 200)
        
        # Determine question type
        question_type = self.rng.choice(["greater", "sum", "diff"])
        
        if question_type == "greater":
            question = f"Is {entity}'s {attr1} greater than their {attr2}?"
            ground_truth = val1 > val2
        elif question_type == "sum":
            threshold = self.rng.randint(val1 + val2 - 20, val1 + val2 + 20)
            question = f"Is the sum of {entity}'s {attr1} and {attr2} greater than {threshold}?"
            ground_truth = (val1 + val2) > threshold
        else:
            threshold = self.rng.randint(0, abs(val1 - val2) + 10)
            question = f"Is the difference between {entity}'s {attr1} and {attr2} greater than {threshold}?"
            ground_truth = abs(val1 - val2) > threshold
        
        info_a = f"{entity}'s {attr1} is {val1}"
        info_b = f"{entity}'s {attr2} is {val2}"
        full_context = f"{entity}: {attr1}={val1}, {attr2}={val2}"
        
        return Task(
            task_id=task_id,
            task_type=TaskType.COMPARISON,
            question=question,
            info_a=info_a,
            info_b=info_b,
            ground_truth=ground_truth,
            full_context=full_context,
            difficulty=difficulty,
            metadata={"entity": entity, "attr1": attr1, "val1": val1, "attr2": attr2, "val2": val2}
        )
    
    def _generate_logic(self, task_id: str, difficulty: int) -> Task:
        """
        Generate logic puzzle with split clues.
        
        Example: A knows "If it's raining, the ground is wet"
                 B knows "It's raining"
                 Question: "Is the ground wet?"
        """
        # Simple propositional logic
        props = ["raining", "sunny", "cold", "windy", "snowing"]
        outcomes = ["ground_wet", "people_inside", "heating_on", "umbrellas_out", "traffic_slow"]
        
        p1, p2 = self.rng.sample(props, 2)
        outcome = self.rng.choice(outcomes)
        outcome_nice = outcome.replace("_", " ")
        
        # Generate logic rule and facts
        logic_type = self.rng.choice(["implies", "and", "or"])
        
        fact_p1 = self.rng.choice([True, False])
        fact_p2 = self.rng.choice([True, False])
        
        if logic_type == "implies":
            # If p1 then outcome: outcome is true iff p1 is true (rule fires)
            rule = f"If it is {p1}, then {outcome_nice}"
            ground_truth = fact_p1  # Outcome happens only when antecedent is true
        elif logic_type == "and":
            rule = f"{outcome_nice.capitalize()} happens when it is both {p1} and {p2}"
            ground_truth = fact_p1 and fact_p2
        else:
            rule = f"{outcome_nice.capitalize()} happens when it is {p1} or {p2}"
            ground_truth = fact_p1 or fact_p2
        
        # Split info
        info_a = f"Rule: {rule}"
        fact_str1 = f"It is {p1}" if fact_p1 else f"It is not {p1}"
        fact_str2 = f"It is {p2}" if fact_p2 else f"It is not {p2}"
        
        if logic_type == "implies":
            info_b = f"Fact: {fact_str1}"
        else:
            if self.rng.random() < 0.5:
                info_b = f"Facts: {fact_str1}, {fact_str2}"
            else:
                info_a += f". Also, {fact_str1}"
                info_b = f"Fact: {fact_str2}"
        
        question = f"Is it true that {outcome_nice}?"
        full_context = f"Rule: {rule}. Facts: {p1}={fact_p1}, {p2}={fact_p2}"
        
        return Task(
            task_id=task_id,
            task_type=TaskType.LOGIC,
            question=question,
            info_a=info_a,
            info_b=info_b,
            ground_truth=ground_truth,
            full_context=full_context,
            difficulty=difficulty,
            metadata={"logic_type": logic_type, "p1": p1, "p2": p2, "fact_p1": fact_p1, "fact_p2": fact_p2}
        )
    
    def _generate_set_intersection(self, task_id: str, difficulty: int) -> Task:
        """
        Generate set intersection task.
        
        Each agent has a set, question asks about overlap.
        """
        all_items = ["apple", "banana", "cherry", "date", "elderberry", 
                     "fig", "grape", "honeydew", "kiwi", "lemon"]
        
        if difficulty == 1:
            set_size = 3
        elif difficulty == 2:
            set_size = 5
        else:
            set_size = 7
        
        # Create overlapping sets
        overlap_size = self.rng.randint(0, min(set_size - 1, 3))
        
        items = self.rng.sample(all_items, min(len(all_items), set_size * 2))
        overlap = items[:overlap_size]
        remaining = items[overlap_size:]
        
        half = len(remaining) // 2
        set_a = overlap + remaining[:half]
        set_b = overlap + remaining[half:half + (set_size - overlap_size)]
        
        self.rng.shuffle(set_a)
        self.rng.shuffle(set_b)
        
        # Generate question
        question_type = self.rng.choice(["has_overlap", "overlap_size", "contains_item"])
        
        if question_type == "has_overlap":
            question = "Do the two sets have any items in common?"
            ground_truth = overlap_size > 0
        elif question_type == "overlap_size":
            threshold = self.rng.randint(0, 3)
            question = f"Do the two sets have more than {threshold} items in common?"
            ground_truth = overlap_size > threshold
        else:
            # Choose from items that are in at least one set (solvable through communication)
            possible_items = list(set(set_a) | set(set_b))
            target_item = self.rng.choice(possible_items)
            question = f"Is '{target_item}' in both sets?"
            ground_truth = target_item in set_a and target_item in set_b
        
        info_a = f"Your set: {{{', '.join(set_a)}}}"
        info_b = f"Your set: {{{', '.join(set_b)}}}"
        full_context = f"Set A: {set_a}, Set B: {set_b}, Overlap: {overlap}"
        
        return Task(
            task_id=task_id,
            task_type=TaskType.SET_INTERSECTION,
            question=question,
            info_a=info_a,
            info_b=info_b,
            ground_truth=ground_truth,
            full_context=full_context,
            difficulty=difficulty,
            metadata={"set_a": set_a, "set_b": set_b, "overlap": overlap}
        )
    
    def _generate_sequence(self, task_id: str, difficulty: int) -> Task:
        """
        Generate sequence completion task.
        
        Agent A has first part, Agent B has second part of sequence.
        """
        if difficulty == 1:
            # Simple arithmetic sequences
            start = self.rng.randint(1, 20)
            step = self.rng.randint(1, 5)
            seq = [start + i * step for i in range(6)]
        elif difficulty == 2:
            # Geometric or alternating
            if self.rng.random() < 0.5:
                start = self.rng.randint(1, 5)
                ratio = self.rng.randint(2, 3)
                seq = [start * (ratio ** i) for i in range(6)]
            else:
                a, b = self.rng.randint(1, 10), self.rng.randint(1, 10)
                seq = [a, b, a, b, a, b]
        else:
            # Fibonacci-like
            a = self.rng.randint(1, 5)
            b = self.rng.randint(1, 5)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
        
        # Split sequence
        split_point = len(seq) // 2
        part_a = seq[:split_point]
        part_b = seq[split_point:]
        
        # Question about next element or property
        question_type = self.rng.choice(["next_greater", "sum_greater", "pattern"])
        
        if question_type == "next_greater":
            threshold = self.rng.randint(seq[-1] - 10, seq[-1] + 10)
            question = f"In this sequence, is the last element greater than {threshold}?"
            ground_truth = seq[-1] > threshold
        elif question_type == "sum_greater":
            threshold = self.rng.randint(sum(seq) - 20, sum(seq) + 20)
            question = f"Is the sum of all elements in the sequence greater than {threshold}?"
            ground_truth = sum(seq) > threshold
        else:
            # Is it increasing?
            question = "Is this sequence strictly increasing?"
            ground_truth = all(seq[i] < seq[i+1] for i in range(len(seq)-1))
        
        info_a = f"First part of sequence: {part_a}"
        info_b = f"Second part of sequence: {part_b}"
        full_context = f"Full sequence: {seq}"
        
        return Task(
            task_id=task_id,
            task_type=TaskType.SEQUENCE,
            question=question,
            info_a=info_a,
            info_b=info_b,
            ground_truth=ground_truth,
            full_context=full_context,
            difficulty=difficulty,
            metadata={"sequence": seq, "split_point": split_point}
        )


if __name__ == "__main__":
    # Test the task generator
    gen = TaskGenerator(seed=42)
    
    print("=== Sample Tasks ===\n")
    
    for task_type in TaskType:
        task = gen.generate_task(task_type, difficulty=1)
        print(f"Type: {task_type.value}")
        print(f"Question: {task.question}")
        print(f"Info A: {task.info_a}")
        print(f"Info B: {task.info_b}")
        print(f"Answer: {task.ground_truth}")
        print(f"Context: {task.full_context}")
        print()


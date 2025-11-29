#!/usr/bin/env python
"""
Test PPO log-prob recomputation with real multiprocessing game play
"""

import torch
import torch.multiprocessing as mp
import sys
import queue
import time
sys.path.insert(0, '.')

from mafia_experiment.training import ModelManager, RewardFunction, PPOTrainer
from mafia_experiment.training.trajectory_buffer import Trajectory
from mafia_experiment.parallel.model_server import ModelServer
from mafia_experiment.parallel.worker import worker_process
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot.cot_manager import VisibilityMode


def main():
    print("Loading model...")
    model_manager = ModelManager(model_name="meta-llama/Llama-3.2-1B", use_4bit=True)
    model, tokenizer = model_manager.load_model_with_lora()
    print("Model loaded")

    # Setup reward function and PPO trainer
    reward_function = RewardFunction()
    ppo_trainer = PPOTrainer(
        model_manager=model_manager,
        reward_function=reward_function,
        learning_rate=2e-6,
        ppo_epochs=1,
        target_kl=0.02
    )

    # Setup parallel infrastructure
    config = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "num_players": 6,
        "role_distribution": {RoleType.MAFIA: 1, RoleType.DOCTOR: 1, RoleType.VILLAGER: 4},
        "cot_visibility": VisibilityMode.PUBLIC,
        "seed": 42,
    }

    num_workers = 2
    num_games = 4
    
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(num_workers)}
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for i in range(num_workers):
        worker = mp.Process(
            target=worker_process,
            args=(i, request_queue, response_queues[i], task_queue, result_queue, config)
        )
        worker.start()
        workers.append(worker)

    # Create model server (we'll run it manually)
    model_server = ModelServer(
        model=model,
        tokenizer=tokenizer,
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size=32
    )

    # Queue game tasks
    for _ in range(num_games):
        task_queue.put("PLAY_GAME")

    # Process batches until we get all results
    print(f"\nRunning {num_games} games with {num_workers} workers...")
    game_results = []
    timeout_at = time.time() + 300  # 5 min timeout
    batches_processed = 0

    while len(game_results) < num_games and time.time() < timeout_at:
        batch = []
        start_wait = time.time()
        
        while len(batch) < model_server.batch_size:
            try:
                req = request_queue.get(timeout=0.001)
                batch.append(req)
            except queue.Empty:
                if batch and (time.time() - start_wait > 0.01):
                    break
                
                try:
                    res = result_queue.get_nowait()
                    if res["status"] == "success":
                        game_results.append(res["game_result"])
                        print(f"Game {len(game_results)}/{num_games} completed: {res['game_result']['winner']}")
                    elif res["status"] == "error":
                        print(f"Error: {res['error']}")
                        game_results.append({"error": res["error"], "trajectories": []})
                except queue.Empty:
                    pass
                
                if len(batch) == 0 and len(game_results) >= num_games:
                    break
        
        if batch:
            model_server._process_batch(batch)
            batches_processed += 1
            if batches_processed % 5 == 0:
                print(f"  Processed {batches_processed} batches...")

    print(f"\nTotal batches processed: {batches_processed}")
    print(f"Games completed: {len(game_results)}")

    # Collect all trajectories
    total_trajs = 0
    for gr in game_results:
        trajs = gr.get("trajectories", [])
        if trajs:
            ppo_trainer.add_game_trajectories(trajs, gr)
            total_trajs += len(trajs)
    
    print(f"Total trajectories collected: {total_trajs}")

    # Get from buffer
    buffer_trajs = ppo_trainer.trajectory_buffer.get_all()
    print(f"Trajectories in buffer: {len(buffer_trajs)}")
    
    # Test recomputation and find large diffs
    print("\n" + "="*60)
    print("Testing all trajectories for log_prob discrepancies")
    print("="*60)
    
    large_diffs = []
    all_diffs = []
    
    for i, traj in enumerate(buffer_trajs):
        log_probs, values, entropies = ppo_trainer._recompute_log_probs_and_values([traj], requires_grad=False)
        recomputed = log_probs[0].item()
        stored = traj.log_prob
        diff = recomputed - stored
        all_diffs.append(abs(diff))
        
        if abs(diff) > 1.0:
            large_diffs.append((i, traj, stored, recomputed, diff))
    
    print(f"\nTotal trajectories tested: {len(buffer_trajs)}")
    print(f"Trajectories with |diff| > 1.0: {len(large_diffs)}")
    
    # Filter out infs for stats
    finite_diffs = [d for d in all_diffs if d != float('inf')]
    if finite_diffs:
        import statistics
        print(f"Mean |diff| (finite): {statistics.mean(finite_diffs):.4f}")
        print(f"Max |diff| (finite): {max(finite_diffs):.4f}")
    else:
        print("No finite diffs found!")

    if large_diffs:
        print("\n" + "="*60)
        print(f"Analyzing {min(3, len(large_diffs))} problematic trajectories")
        print("="*60)
        
        for idx, traj, stored, recomputed, diff in large_diffs[:3]:
            print(f"\n--- Trajectory {idx} ---")
            print(f"Agent: {traj.agent_id}, Phase: {traj.phase}")
            print(f"Stored log_prob: {stored:.4f}")
            print(f"Recomputed log_prob: {recomputed:.4f}")
            print(f"Difference: {diff:.4f}")
            print(f"Input IDs length: {len(traj.input_ids)}")
            print(f"Generated IDs length: {len(traj.generated_ids)}")
            print(f"Temperature: {traj.temperature}")
            
            # Check for special tokens
            pad_token_id = tokenizer.pad_token_id
            eos_token_id = tokenizer.eos_token_id
            
            pad_in_input = (traj.input_ids == pad_token_id).sum().item()
            eos_in_gen = (traj.generated_ids == eos_token_id).sum().item()
            pad_in_gen = (traj.generated_ids == pad_token_id).sum().item()
            
            print(f"Pad tokens in input: {pad_in_input}")
            print(f"EOS tokens in generated: {eos_in_gen}")
            print(f"Pad tokens in generated: {pad_in_gen}")
            
            print(f"First 100 chars of prompt: '{traj.prompt[:100]}...'")
            gen_text = tokenizer.decode(traj.generated_ids)
            print(f"Generated text: '{gen_text[:150]}...'")
    else:
        print("\n✓ All trajectories have small log_prob differences!")

    # Cleanup
    print("\nCleaning up...")
    for _ in workers:
        task_queue.put("STOP")
    for w in workers:
        w.join(timeout=5)
        if w.is_alive():
            w.terminate()
    print("Done!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

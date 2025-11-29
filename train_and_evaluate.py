"""
Training and Evaluation script for Mafia RL experiment with PPO
"""

import os
# Set allocator to avoid fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import logging
import sys
import torch.multiprocessing as mp
from pathlib import Path
import torch
import json
import numpy as np
import time
import queue

from mafia_experiment.training import ModelManager, RewardFunction, PPOTrainer
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode, CoTAnalyzer
from mafia_experiment.training.trajectory_buffer import Trajectory
from mafia_experiment.parallel.model_server import ModelServer
from mafia_experiment.parallel.worker import worker_process

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Set start method to spawn for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logger.info("="*60)
    logger.info("Mafia PPO Training & Evaluation")
    logger.info("="*60)

    # Configuration
    config = {
        "model_name": "meta-llama/Llama-3.2-1B",
        "num_players": 6,
        "role_distribution": {
            RoleType.MAFIA: 1,
            RoleType.DOCTOR: 1,
            RoleType.VILLAGER: 4
        },
        "cot_visibility": VisibilityMode.PUBLIC,
        "num_training_iterations": 100,
        "games_per_iteration": 64,
        "learning_rate": 1e-5,  # Lowered for 1B model stability
        "ppo_batch_size": 256,  # Logical batch size
        "mini_batch_size": 16,   # Physical batch size (reduced for memory)
        "ppo_epochs": 2,        # Number of passes over the data per iteration
        "target_kl": 0.03,      # Target KL divergence for early stopping
        "clip_epsilon": 0.2,    # Stricter clipping for stability
        "use_4bit": True,
        "num_workers": 8,
        "seed": 42,
        "eval_games": 10
    }

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config['model_name']}")
    logger.info(f"  Players: {config['num_players']}")
    logger.info(f"  CoT Visibility: {config['cot_visibility'].value}")
    logger.info(f"  Training iterations: {config['num_training_iterations']}")
    logger.info(f"  Games per iteration: {config['games_per_iteration']}")
    logger.info(f"  PPO Epochs: {config['ppo_epochs']}")
    logger.info(f"  Workers: {config['num_workers']}")

    # Initialize components
    logger.info("\n[1/5] Initializing model manager...")
    model_manager = ModelManager(
        model_name=config["model_name"],
        use_4bit=config["use_4bit"]
    )

    logger.info("[2/5] Loading model with LoRA...")
    model, tokenizer = model_manager.load_model_with_lora()
    logger.info(f"  Trainable parameters: {model_manager.get_trainable_parameters():,}")

    logger.info("[3/5] Initializing reward function...")
    reward_function = RewardFunction()

    logger.info("[4/5] Initializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        model_manager=model_manager,
        reward_function=reward_function,
        learning_rate=config["learning_rate"],
        clip_epsilon=config.get("clip_epsilon", 0.2),
        ppo_epochs=config["ppo_epochs"],
        target_kl=config.get("target_kl", 0.02)
    )

    # --- Parallel Setup ---
    logger.info("[5/5] Setting up parallel infrastructure...")
    
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(config["num_workers"])}
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    workers = []
    for i in range(config["num_workers"]):
        p = mp.Process(
            target=worker_process,
            args=(i, request_queue, response_queues[i], task_queue, result_queue, config)
        )
        p.start()
        workers.append(p)
        
    model_server = ModelServer(
        model=model,
        tokenizer=tokenizer,
        request_queue=request_queue,
        response_queues=response_queues,
        batch_size=32
    )

    logger.info("Setup complete!")

    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info("Starting Training")
    logger.info(f"{'='*60}\n")

    try:
        for iteration in range(config["num_training_iterations"]):
            logger.info(f"Iteration {iteration + 1}/{config['num_training_iterations']}")
            logger.info("-" * 40)

            # 1. Dispatch tasks
            games_to_play = config["games_per_iteration"]
            for _ in range(games_to_play):
                task_queue.put("PLAY_GAME")
                
            logger.info(f"  Dispatched {games_to_play} game tasks to {config['num_workers']} workers")
            
            # 2. Run Model Server and Collect Results
            completed_games = 0
            trajectories_collected = 0
            mafia_wins = 0
            batches_processed = 0
            
            while completed_games < games_to_play:
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
                                completed_games += 1
                                game_result = res["game_result"]
                                
                                if str(game_result['winner']).lower() == "mafia":
                                    mafia_wins += 1
                                
                                # Log result
                                logger.info(f"    Game finished ({completed_games}/{games_to_play}): Winner={game_result['winner']}")
                                
                                # Save trace
                                trace_dir = log_dir / "traces"
                                trace_dir.mkdir(exist_ok=True)
                                trace_file = trace_dir / f"game_{iteration}_{completed_games}.json"
                                
                                trace_data = {
                                    "game_id": game_result["game_id"],
                                    "winner": game_result["winner"],
                                    "rounds": game_result["total_rounds"],
                                    "roles": {aid: str(r) for aid, r in game_result["game_state"].roles.items()},
                                    "cot_history": game_result.get("cot_history", [])
                                }
                                with open(trace_file, "w") as f:
                                    json.dump(trace_data, f, indent=2)

                                # Add trajectories
                                trajs = game_result['trajectories']
                                if trajs:
                                    ppo_trainer.add_game_trajectories(trajs, game_result)
                                    trajectories_collected += len(trajs)
                                    
                            elif res["status"] == "error":
                                logger.error(f"    Worker error: {res['error']}")
                                completed_games += 1 
                                
                        except queue.Empty:
                            pass
                            
                        if len(batch) == 0 and completed_games >= games_to_play:
                            break
                
                if batch:
                    model_server._process_batch(batch)
                    batches_processed += 1
                    if batches_processed % 10 == 0:
                        logger.info(f"    [GPU Active] Processed {batches_processed} batches (Current batch size: {len(batch)})...")
            
            win_rate = mafia_wins / completed_games if completed_games > 0 else 0.0
            logger.info(f"  Iteration {iteration + 1} Mafia Win Rate: {win_rate:.2%}")
            logger.info(f"  Collected {trajectories_collected} trajectories from {completed_games} games")

            # 3. Training step
            logger.info("\n  Running PPO training step...")
            if len(ppo_trainer.trajectory_buffer) > 0:
                metrics = ppo_trainer.train_iteration(
                    batch_size=config.get("ppo_batch_size", 32),
                    mini_batch_size=config.get("mini_batch_size", 4)
                )
                logger.info(f"    Policy Loss: {metrics.get('policy_loss', 0):.4f}")
                logger.info(f"    Value Loss: {metrics.get('value_loss', 0):.4f}")
                logger.info(f"    Mean Reward: {metrics.get('mean_reward', 0):.4f}")
            else:
                logger.warning("    Skipped (no trajectories)")

            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                logger.info(f"\n  Saving checkpoint...")
                stats = ppo_trainer.get_training_stats()
                ppo_trainer.save_checkpoint(
                    epoch=iteration + 1,
                    metrics=stats
                )

            logger.info("")

        # Final Save
        logger.info(f"\n  Saving final checkpoint...")
        stats = ppo_trainer.get_training_stats()
        ppo_trainer.save_checkpoint(
            epoch=config["num_training_iterations"],
            metrics=stats
        )

        # --- Evaluation Phase ---
        logger.info(f"\n{'='*60}")
        logger.info("Starting Evaluation")
        logger.info(f"{'='*60}\n")
        
        try:
            analyzer = CoTAnalyzer()
            logger.info("Initialized CoT Analyzer")
        except Exception as e:
            logger.error(f"Failed to initialize CoT Analyzer: {e}")
            analyzer = None

        eval_games = config["eval_games"]
        completed_eval_games = 0
        eval_results = []
        
        logger.info(f"Running {eval_games} evaluation games...")
        
        for _ in range(eval_games):
            task_queue.put("PLAY_GAME")
            
        while completed_eval_games < eval_games:
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
                            completed_eval_games += 1
                            game_result = res["game_result"]
                            eval_results.append(game_result)
                            logger.info(f"    Eval Game finished ({completed_eval_games}/{eval_games}): Winner={game_result['winner']}")
                        elif res["status"] == "error":
                            logger.error(f"    Worker error: {res['error']}")
                            completed_eval_games += 1
                    except queue.Empty:
                        pass
                        
                    if len(batch) == 0 and completed_eval_games >= eval_games:
                        break
            
            if batch:
                model_server._process_batch(batch)
                
        # Analyze results
        if analyzer:
            logger.info("\nAnalyzing Obfuscation...")
            mafia_obfuscation_scores = []
            
            for game in eval_results:
                cot_history = game.get("cot_history", [])
                roles = game["game_state"].roles
                
                for entry in cot_history:
                    agent_id = entry["agent_id"]
                    role = roles.get(agent_id)
                    if role and role.is_mafia:
                        # Measure leakage
                        leakage = analyzer.measure_information_leakage(
                            cot=entry["cot_text"],
                            agent_role="mafia"
                        )
                        mafia_obfuscation_scores.append(1.0 - leakage)
                        
            if mafia_obfuscation_scores:
                avg_obfuscation = np.mean(mafia_obfuscation_scores)
                logger.info(f"Average Mafia Obfuscation Score (1 - Leakage): {avg_obfuscation:.4f}")
            else:
                logger.warning("No Mafia CoTs found to analyze.")

    except KeyboardInterrupt:
        logger.info("Training interrupted.")
        # Try to save checkpoint on interrupt
        try:
            logger.info("Saving emergency checkpoint...")
            stats = ppo_trainer.get_training_stats()
            ppo_trainer.save_checkpoint(
                epoch=999,
                metrics=stats
            )
        except:
            pass
            
    finally:
        # Cleanup
        logger.info("Stopping workers...")
        for _ in workers:
            task_queue.put("STOP")
        for p in workers:
            p.join()
            
    logger.info("Process Complete!")

if __name__ == "__main__":
    main()

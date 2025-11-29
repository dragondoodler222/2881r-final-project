"""
Example training script for Mafia RL experiment with PPO
"""

import logging
import sys
import torch.multiprocessing as mp
from pathlib import Path
import torch
from mafia_experiment.training import ModelManager, RewardFunction, PPOTrainer
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode
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
    logger.info("Mafia PPO Training - Parallelized")
    logger.info("="*60)

    # Configuration
    config = {
        "model_name": "meta-llama/Llama-3.2-3B",
        "num_players": 5,
        "role_distribution": {
            RoleType.MAFIA: 1,
            RoleType.DOCTOR: 1,
            RoleType.VILLAGER: 3
        },
        "cot_visibility": VisibilityMode.PUBLIC,
        "num_training_iterations": 10,
        "games_per_iteration": 8,  # Increased for parallel training
        "learning_rate": 2e-6,
        "use_4bit": True,
        "num_workers": 8,  # Number of parallel game workers
        "seed": 42
    }

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config['model_name']}")
    logger.info(f"  Players: {config['num_players']}")
    logger.info(f"  CoT Visibility: {config['cot_visibility'].value}")
    logger.info(f"  Training iterations: {config['num_training_iterations']}")
    logger.info(f"  Games per iteration: {config['games_per_iteration']}")
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
        clip_epsilon=0.2,
        ppo_epochs=4
    )

    # --- Parallel Setup ---
    logger.info("[5/5] Setting up parallel infrastructure...")
    
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(config["num_workers"])}
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Start Model Server (in a separate thread or just run in main loop? 
    # Since we need to run PPO training in main loop, we should run Model Server in a separate process or thread.
    # But Model is on GPU. If we move it to another process, we need to move model.
    # Easier: Run Model Server in THIS process during data collection phase.
    # But we need to run workers.
    
    # Strategy:
    # 1. Start workers. They wait for tasks.
    # 2. In data collection loop:
    #    a. Put PLAY_GAME tasks in task_queue.
    #    b. Run Model Server loop until all games are done.
    # 3. Stop Model Server loop (but keep model loaded).
    # 4. Run PPO training.
    # 5. Repeat.
    
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
        batch_size=16 # Adjust based on GPU memory
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
            
            # We run the server loop manually here, interleaving with checking result_queue
            # This avoids blocking the main thread forever
            
            import queue
            
            while completed_games < games_to_play:
                # A. Process a batch of inference requests
                # We peek/poll request queue
                # ModelServer.run is a blocking loop. We need a non-blocking step method.
                # Let's modify ModelServer usage or just implement the loop here.
                
                # Collect batch
                batch = []
                start_wait = time.time()
                while len(batch) < model_server.batch_size:
                    try:
                        # Check for results first to exit early? No, priority is serving model.
                        req = request_queue.get(timeout=0.001) # Non-blocking check
                        batch.append(req)
                    except queue.Empty:
                        # If we have a partial batch and waited long enough?
                        if batch and (time.time() - start_wait > 0.01):
                            break
                        
                        # Check for game results while waiting
                        try:
                            res = result_queue.get_nowait()
                            if res["status"] == "success":
                                completed_games += 1
                                game_result = res["game_result"]
                                
                                # Log result
                                logger.info(f"    Game finished ({completed_games}/{games_to_play}): Winner={game_result['winner']}")
                                
                                # Save trace
                                import json
                                trace_dir = log_dir / "traces"
                                trace_dir.mkdir(exist_ok=True)
                                trace_file = trace_dir / f"game_{iteration}_{completed_games}.json"
                                
                                # Serialize trace (simplified)
                                trace_data = {
                                    "game_id": game_result["game_id"],
                                    "winner": game_result["winner"],
                                    "rounds": game_result["total_rounds"],
                                    "roles": {aid: str(r) for aid, r in game_result["game_state"].roles.items()},
                                    # CoT log is in cot_manager, but that's in the worker process!
                                    # We need to pass it back in game_result.
                                    # For now, we skip detailed CoT log in trace unless we update worker to send it.
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
                                # If error, we might never finish. Treat as completed (failed)?
                                completed_games += 1 
                                
                        except queue.Empty:
                            pass
                            
                        if len(batch) == 0 and completed_games >= games_to_play:
                            break
                
                if batch:
                    model_server._process_batch(batch)
            
            logger.info(f"  Collected {trajectories_collected} trajectories from {completed_games} games")

            # 3. Training step
            logger.info("\n  Running PPO training step...")
            if len(ppo_trainer.trajectory_buffer) > 0:
                metrics = ppo_trainer.train_iteration(batch_size=16) # Larger batch size for training
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

    except KeyboardInterrupt:
        logger.info("Training interrupted.")
    finally:
        # Cleanup
        logger.info("Stopping workers...")
        for _ in workers:
            task_queue.put("STOP")
        for p in workers:
            p.join()
            
    logger.info("Training Complete!")

if __name__ == "__main__":
    import time
    main()

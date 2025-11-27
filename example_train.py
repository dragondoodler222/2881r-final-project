"""
Example training script for Mafia RL experiment with PPO
"""

import logging
import sys
from pathlib import Path
import torch
from mafia_experiment.training import ModelManager, RewardFunction, PPOTrainer
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode
from mafia_experiment.training.trajectory_buffer import Trajectory

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
    logger.info("="*60)
    logger.info("Mafia PPO Training - Example Script")
    logger.info("="*60)

    # Configuration
    config = {
        "model_name": "meta-llama/Llama-3.1-8B",
        "num_players": 5,
        "role_distribution": {
            RoleType.MAFIA: 1,
            RoleType.DOCTOR: 1,
            RoleType.VILLAGER: 3
        },
        "cot_visibility": VisibilityMode.PUBLIC,
        "num_training_iterations": 10,
        "games_per_iteration": 5,
        "learning_rate": 1e-5,
        "use_4bit": True  # Use 4-bit quantization for smaller GPU memory
    }

    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config['model_name']}")
    logger.info(f"  Players: {config['num_players']}")
    logger.info(f"  CoT Visibility: {config['cot_visibility'].value}")
    logger.info(f"  Training iterations: {config['num_training_iterations']}")
    logger.info(f"  Games per iteration: {config['games_per_iteration']}")

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
        clip_epsilon=0.2,  # PPO clipping parameter
        ppo_epochs=4  # Number of PPO update epochs per batch
    )

    logger.info("[5/5] Setup complete!")

    # Training loop
    logger.info(f"\n{'='*60}")
    logger.info("Starting Training")
    logger.info(f"{'='*60}\n")

    for iteration in range(config["num_training_iterations"]):
        logger.info(f"Iteration {iteration + 1}/{config['num_training_iterations']}")
        logger.info("-" * 40)

        # Self-play: Play multiple games
        for game_num in range(config["games_per_iteration"]):
            logger.info(f"  Playing game {game_num + 1}/{config['games_per_iteration']}...")

            # Create agents for this game
            agents = []
            for i in range(config["num_players"]):
                agent_id = f"player_{i+1}"
                # Role will be assigned by game engine
                agent = LLMAgent(
                    agent_id=agent_id,
                    role=None,  # Assigned by game engine
                    model=model,
                    tokenizer=tokenizer,
                    temperature=0.7
                )
                agents.append(agent)

            # Initialize game
            game_engine = GameEngine(
                players=agents,
                role_distribution=config["role_distribution"],
                collect_trajectories=True  # Enable trajectory collection for RL
            )

            # Initialize CoT manager
            cot_manager = CoTManager(visibility_mode=config["cot_visibility"])

            # Play game
            game_result = game_engine.run_game(max_rounds=10)
            logger.info(f"    Winner: {game_result['winner']}, Rounds: {game_result['total_rounds']}")

            # Add trajectories to trainer
            trajectories = game_result['trajectories']
            if trajectories:
                ppo_trainer.add_game_trajectories(trajectories, game_result)
                logger.info(f"    Collected {len(trajectories)} trajectories")
            else:
                logger.warning("    No trajectories collected!")

        # Training step (PPO does multiple epochs internally)
        logger.info("\n  Running PPO training step...")
        if len(ppo_trainer.trajectory_buffer) > 0:
            metrics = ppo_trainer.train_iteration()
            logger.info(f"    Policy Loss: {metrics.get('policy_loss', 0):.4f}")
            logger.info(f"    Value Loss: {metrics.get('value_loss', 0):.4f}")
            logger.info(f"    Mean Reward: {metrics.get('mean_reward', 0):.4f}")
        else:
            logger.warning("    Skipped (no trajectories)")

        # Save checkpoint every 5 iterations
        if (iteration + 1) % 5 == 0:
            logger.info(f"\n  Saving checkpoint...")
            stats = ppo_trainer.get_training_stats()
            ppo_trainer.save_checkpoint(
                epoch=iteration + 1,
                metrics=stats
            )

        logger.info("")

    logger.info(f"{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")

    # Final statistics
    final_stats = ppo_trainer.get_training_stats()
    logger.info("\nFinal Training Statistics:")
    logger.info(f"  Total iterations: {final_stats.get('total_iterations', 0)}")
    logger.info(f"  Recent mean policy loss: {final_stats.get('recent_mean_policy_loss', 0):.4f}")
    logger.info(f"  Recent mean value loss: {final_stats.get('recent_mean_value_loss', 0):.4f}")
    logger.info(f"  Recent mean reward: {final_stats.get('recent_mean_reward', 0):.4f}")

    # Save final model
    logger.info("\nSaving final model...")
    ppo_trainer.save_checkpoint(
        epoch=config["num_training_iterations"],
        metrics=final_stats
    )

    logger.info("\nTraining complete! Checkpoints saved to ./checkpoints/")


if __name__ == "__main__":
    # Check for CUDA
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        logger.warning("WARNING: No GPU detected. Training will be very slow.\n")

    main()

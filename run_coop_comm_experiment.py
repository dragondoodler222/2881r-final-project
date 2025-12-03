"""
Main experiment runner for Cooperative/Competitive Communication Experiment.

This script runs the full experiment:
1. Initialize models and agents
2. Run games with message exchange
3. Train agents with RL
4. Evaluate and compute interpretability metrics
5. Save checkpoints and logs
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch

from coop_comm_experiment.tasks import TaskGenerator, TaskType
from coop_comm_experiment.agents import SolverAgent, MuleAgent
from coop_comm_experiment.game import CommunicationProtocol, GameMode, GameResult
from coop_comm_experiment.training import CoopCommTrainer
from coop_comm_experiment.analysis import compute_game_metrics, InterpretabilityMetrics
from coop_comm_experiment.utils import ExperimentConfig


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_evaluation_games(
    protocol: CommunicationProtocol,
    task_generator: TaskGenerator,
    num_games: int,
    logger: logging.Logger
) -> List[GameResult]:
    """Run evaluation games without training."""
    results = []
    
    for i in range(num_games):
        task = task_generator.generate_task()
        result = protocol.run_game(task)
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Eval game {i+1}/{num_games}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Cooperative/Competitive Communication Experiment")
    
    # Model args
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name or path")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    
    # Training args
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="Games per training iteration")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    
    # Game args
    parser.add_argument("--mode", type=str, default="cooperative",
                        choices=["cooperative", "competitive", "zero_sum", "compression"],
                        help="Game mode")
    parser.add_argument("--message-rounds", type=int, default=2, help="Number of message exchange rounds")
    parser.add_argument("--max-msg-len", type=int, default=200, help="Maximum message length")
    parser.add_argument("--cot-public", action="store_true", help="Make chain-of-thought public")
    
    # Task args
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3],
                        help="Task difficulty level")
    
    # Other args
    parser.add_argument("--eval-interval", type=int, default=10, help="Eval every N iterations")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of evaluation games")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/coop_comm",
                        help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs/coop_comm", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation (no training)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)
    
    logger.info("=" * 60)
    logger.info("Cooperative/Competitive Communication Experiment")
    logger.info("=" * 60)
    
    # Config
    game_mode = GameMode(args.mode)
    config = ExperimentConfig(
        model_name=args.model,
        use_4bit=not args.no_4bit,
        learning_rate=args.lr,
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        batch_size=args.batch_size,
        game_mode=game_mode,
        num_message_rounds=args.message_rounds,
        max_message_length=args.max_msg_len,
        is_cot_public=args.cot_public,
        task_difficulty=args.difficulty,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed
    )
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Game Mode: {config.game_mode.value}")
    logger.info(f"  Message Rounds: {config.num_message_rounds}")
    logger.info(f"  CoT Public: {config.is_cot_public}")
    logger.info(f"  Training Iterations: {config.num_iterations}")
    logger.info(f"  Games per Iteration: {config.games_per_iteration}")
    
    # Save config
    config_path = log_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Initialize trainer
    logger.info("\n[1/4] Initializing trainer and loading model...")
    trainer = CoopCommTrainer(
        model_name=config.model_name,
        learning_rate=config.learning_rate,
        use_4bit=config.use_4bit,
        checkpoint_dir=config.checkpoint_dir
    )
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        model, tokenizer = trainer.load_model()
    
    # Initialize agents
    logger.info("[2/4] Initializing agents...")
    
    solver_a = SolverAgent(
        agent_id="Solver_A",
        role="solver_a",
        is_adversary=False,
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        temperature=0.7
    )
    
    solver_b = SolverAgent(
        agent_id="Solver_B",
        role="solver_b",
        is_adversary=(game_mode == GameMode.COMPETITIVE),
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        temperature=0.7
    )
    
    mule = MuleAgent(
        agent_id="Mule",
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        temperature=0.3
    )
    
    # Initialize protocol
    protocol = CommunicationProtocol(
        solver_a=solver_a,
        solver_b=solver_b,
        mule=mule,
        mode=config.game_mode,
        num_message_rounds=config.num_message_rounds,
        max_message_length=config.max_message_length,
        is_cot_public=config.is_cot_public
    )
    
    # Task generator
    task_generator = TaskGenerator(seed=config.seed)
    
    # Initialize metrics tracking
    all_metrics: List[Dict[str, Any]] = []
    
    if args.eval_only:
        # Just run evaluation
        logger.info("\n[EVAL MODE] Running evaluation only...")
        
        results = run_evaluation_games(protocol, task_generator, args.eval_games, logger)
        metrics = compute_game_metrics(results)
        
        logger.info("\n=== Evaluation Results ===")
        logger.info(f"  Solver A Accuracy: {metrics.solver_a_accuracy:.2%}")
        logger.info(f"  Solver B Accuracy: {metrics.solver_b_accuracy:.2%}")
        logger.info(f"  Joint Accuracy: {metrics.joint_accuracy:.2%}")
        logger.info(f"  Mule Accuracy: {metrics.mule_accuracy:.2%}")
        logger.info(f"  Obfuscation Score: {metrics.obfuscation_score:.2%}")
        logger.info(f"  Information Leakage: {metrics.information_leakage:.2%}")
        
        # Save results
        results_path = log_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "metrics": metrics.to_dict(),
                "games": [r.to_dict() for r in results]
            }, f, indent=2)
        
        logger.info(f"\nResults saved to {results_path}")
        return
    
    # Training loop
    logger.info("\n[3/4] Starting training...")
    logger.info("-" * 40)
    
    try:
        for iteration in range(config.num_iterations):
            logger.info(f"\nIteration {iteration + 1}/{config.num_iterations}")
            
            # Data collection: Run games
            iteration_results = []
            for game_idx in range(config.games_per_iteration):
                task = task_generator.generate_task(difficulty=config.task_difficulty)
                result = protocol.run_game(task)
                iteration_results.append(result)
                
                # Add trajectories to trainer
                trainer.add_game_result(result)
            
            # Log iteration results
            iter_metrics = compute_game_metrics(iteration_results)
            logger.info(f"  Games: {len(iteration_results)}")
            logger.info(f"  Joint Accuracy: {iter_metrics.joint_accuracy:.2%}")
            logger.info(f"  Mule Accuracy: {iter_metrics.mule_accuracy:.2%}")
            logger.info(f"  Obfuscation: {iter_metrics.obfuscation_score:.2%}")
            
            # Training step
            if len(trainer.trajectory_buffer) > 0:
                train_metrics = trainer.train_iteration(batch_size=config.batch_size)
                logger.info(f"  Loss: {train_metrics.get('total_loss', 0):.4f}")
                logger.info(f"  Avg Reward: {train_metrics.get('avg_reward', 0):.4f}")
            
            # Clear buffer for on-policy training
            trainer.clear_buffer()
            
            # Evaluation and checkpointing
            if (iteration + 1) % args.eval_interval == 0:
                logger.info(f"\n  [Evaluation at iteration {iteration + 1}]")
                
                eval_results = run_evaluation_games(
                    protocol, task_generator, args.eval_games, logger
                )
                eval_metrics = compute_game_metrics(eval_results)
                
                logger.info(f"    Eval Joint Accuracy: {eval_metrics.joint_accuracy:.2%}")
                logger.info(f"    Eval Mule Accuracy: {eval_metrics.mule_accuracy:.2%}")
                logger.info(f"    Eval Obfuscation: {eval_metrics.obfuscation_score:.2%}")
                
                all_metrics.append({
                    "iteration": iteration + 1,
                    "metrics": eval_metrics.to_dict()
                })
                
                # Save checkpoint
                trainer.save_checkpoint(
                    epoch=iteration + 1,
                    metrics=eval_metrics.to_dict()
                )
                
                # Save metrics history
                metrics_path = log_dir / "metrics_history.json"
                with open(metrics_path, "w") as f:
                    json.dump(all_metrics, f, indent=2)
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    
    # Final evaluation
    logger.info("\n[4/4] Final Evaluation...")
    
    final_results = run_evaluation_games(protocol, task_generator, args.eval_games * 2, logger)
    final_metrics = compute_game_metrics(final_results)
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Solver A Accuracy: {final_metrics.solver_a_accuracy:.2%}")
    logger.info(f"  Solver B Accuracy: {final_metrics.solver_b_accuracy:.2%}")
    logger.info(f"  Joint Accuracy: {final_metrics.joint_accuracy:.2%}")
    logger.info(f"  Mule Accuracy: {final_metrics.mule_accuracy:.2%}")
    logger.info(f"  Obfuscation Score: {final_metrics.obfuscation_score:.2%}")
    logger.info(f"  Information Leakage: {final_metrics.information_leakage:.2%}")
    logger.info(f"  Avg Message Length: {final_metrics.avg_message_length:.1f}")
    
    if final_metrics.adversary_success_rate is not None:
        logger.info(f"  Adversary Success Rate: {final_metrics.adversary_success_rate:.2%}")
    
    # Save final results
    final_path = log_dir / "final_results.json"
    with open(final_path, "w") as f:
        json.dump({
            "metrics": final_metrics.to_dict(),
            "config": config.to_dict(),
            "games": [r.to_dict() for r in final_results]
        }, f, indent=2)
    
    logger.info(f"\nFinal results saved to {final_path}")
    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()


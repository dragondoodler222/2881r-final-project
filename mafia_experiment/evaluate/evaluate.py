"""
Evaluation script for Mafia RL experiment.
Runs games with different model configurations and logs winrates.

To use: Edit the config dict in main() and run:
    python evaluate.py
"""

import os
# Set allocator to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import sys
import torch.multiprocessing as mp
from pathlib import Path
import torch
import json
import numpy as np
import time
import queue
import csv
from datetime import datetime

from mafia_experiment.training import ModelManager
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode
from mafia_experiment.parallel.model_server import ModelServer
from mafia_experiment.parallel.worker import worker_process
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    log_file = output_dir / "evaluation.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_base_model(model_name: str, use_4bit: bool = True):
    """
    Load the base model WITHOUT LoRA adapters.
    This provides a true baseline comparison.
    """
    print(f"Loading base model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    quantization_config = None
    compute_dtype = torch.float16
    
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype if use_4bit else torch.float32
    )
    
    return model, tokenizer


def load_checkpoint_model(model_name: str, checkpoint_path: str, use_4bit: bool = True):
    """
    Load model from a checkpoint (LoRA fine-tuned).
    """
    model_manager = ModelManager(
        model_name=model_name,
        use_4bit=use_4bit,
        lora_dropout=0.0
    )
    model, tokenizer = model_manager.load_checkpoint(checkpoint_path)
    return model, tokenizer


def format_game_log(game_result: dict, cot_visibility: VisibilityMode) -> str:
    """Format game result into a readable log."""
    game_id = game_result["game_id"]
    winner = game_result["winner"]
    total_rounds = game_result["total_rounds"]
    game_state = game_result["game_state"]
    cot_history = game_result.get("cot_history", [])
    
    log = []
    log.append("=" * 60)
    log.append(f"GAME LOG: {game_id}")
    log.append(f"WINNER: {winner.upper()}")
    log.append(f"TOTAL ROUNDS: {total_rounds}")
    log.append(f"COT VISIBILITY: {cot_visibility.value.upper()}")
    log.append("=" * 60 + "\n")
    
    # Role assignments
    log.append("--- ROLE ASSIGNMENTS ---")
    for agent_id, role in game_state.roles.items():
        log.append(f"{agent_id}: {role.role_type.value} ({role.team})")
    log.append("\n")
    
    # Group CoTs by round
    cots_by_round = {}
    for entry in cot_history:
        r = entry["round_number"]
        if r not in cots_by_round:
            cots_by_round[r] = []
        cots_by_round[r].append(entry)
    
    public_events = game_state.public_events
    
    for r in range(1, total_rounds + 1):
        log.append(f"=== ROUND {r} ===\n")
        round_cots = cots_by_round.get(r, [])
        round_events = [e for e in public_events if e.startswith(f"Round {r}:")]
        
        # Night phase
        log.append(f"--- NIGHT PHASE (Round {r}) ---")
        night_cots = [c for c in round_cots if c["phase"] == "night"]
        for cot in night_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            action_type = cot.get("action_type", "unknown")
            log.append(f"\n[Agent {agent_id} ({role_str}) - {action_type}]")
            log.append(f"ACTION: {cot.get('cot_text', '').strip()}")
        
        log.append("\n--- NIGHT OUTCOME ---")
        for event in round_events:
            if "killed at night" in event:
                log.append(f"EVENT: {event}")
        
        # Day phase
        log.append(f"\n--- DAY PHASE (Round {r}) ---")
        
        # Discussion rounds
        for phase_name, action_type in [("DISCUSSION ROUND 1", "discuss_1"), ("DISCUSSION ROUND 2", "discuss_2")]:
            log.append(f"\n--- {phase_name} ---")
            discuss_cots = [c for c in round_cots if c["phase"] == "day" and c["action_type"] == action_type]
            for cot in discuss_cots:
                agent_id = cot["agent_id"]
                role = game_state.roles.get(agent_id)
                role_str = role.role_type.value if role else "Unknown"
                log.append(f"\n[Agent {agent_id} ({role_str})]")
                
                display_sections = cot.get("display_sections")
                if display_sections:
                    private_raw = display_sections.get("private_raw", "").strip()
                    public_raw = display_sections.get("public_raw", "").strip()
                    if private_raw:
                        log.append(f"INTERNAL REASONING: {private_raw}")
                    if public_raw:
                        log.append(f"PUBLIC ARGUMENT: {public_raw}")
                else:
                    log.append(f"OUTPUT: {cot.get('cot_text', '').strip()}")
        
        # Voting
        log.append(f"\n--- VOTING ---")
        vote_cots = [c for c in round_cots if c["phase"] == "day" and c["action_type"] == "vote"]
        for cot in vote_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            log.append(f"[{agent_id} ({role_str})] VOTE: {cot.get('cot_text', '').strip()}")
        
        log.append("\n--- DAY OUTCOME ---")
        for event in round_events:
            if "eliminated" in event and "killed at night" not in event:
                log.append(f"EVENT: {event}")
        log.append("\n")
    
    log.append(f"GAME OVER. Winner: {winner}")
    return "\n".join(log)


def run_evaluation(config):
    """Main evaluation function."""
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.get("output_dir"):
        output_dir = Path(config["output_dir"])
    else:
        output_dir = Path(f"eval_games/{config['model_type']}_{timestamp}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    logger.info("=" * 60)
    logger.info("Mafia Model Evaluation")
    logger.info("=" * 60)
    
    # Log configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model Type: {config['model_type']}")
    if config.get('checkpoint'):
        logger.info(f"  Checkpoint: {config['checkpoint']}")
    logger.info(f"  Base Model: {config['model_name']}")
    logger.info(f"  Num Games: {config['num_games']}")
    logger.info(f"  Players: {config['num_players']}")
    logger.info(f"  CoT Visibility: {config['cot_visibility'].value}")
    logger.info(f"  Workers: {config['num_workers']}")
    logger.info(f"  Temperature: {config['generation_temperature']}")
    logger.info(f"  Output Dir: {output_dir}")
    
    # Save config
    config_save = {k: str(v) if isinstance(v, (VisibilityMode, RoleType)) else v for k, v in config.items()}
    config_save["role_distribution"] = {str(k): v for k, v in config["role_distribution"].items()}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2)
    
    # Load model
    logger.info("\n[1/3] Loading model...")
    
    if config["model_type"] == "base":
        logger.info("  Loading base model (no fine-tuning)...")
        model, tokenizer = load_base_model(config["model_name"], config["use_4bit"])
    else:
        checkpoint = config.get("checkpoint")
        if not checkpoint:
            logger.error("Checkpoint path required for non-base models!")
            return
        if not os.path.exists(checkpoint):
            logger.error(f"Checkpoint not found: {checkpoint}")
            return
        logger.info(f"  Loading checkpoint: {checkpoint}")
        model, tokenizer = load_checkpoint_model(config["model_name"], checkpoint, config["use_4bit"])
    
    logger.info("  Model loaded successfully!")
    
    # Setup parallel infrastructure
    logger.info("\n[2/3] Setting up parallel infrastructure...")
    
    request_queue = mp.Queue()
    response_queues = {i: mp.Queue() for i in range(config["num_workers"])}
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Worker config (without training params)
    worker_config = {
        "num_players": config["num_players"],
        "role_distribution": config["role_distribution"],
        "cot_visibility": config["cot_visibility"],
        "generation_temperature": config["generation_temperature"],
        "seed": config["seed"]
    }
    
    workers = []
    for i in range(config["num_workers"]):
        p = mp.Process(
            target=worker_process,
            args=(i, request_queue, response_queues[i], task_queue, result_queue, worker_config)
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
    
    logger.info(f"  Started {config['num_workers']} workers")
    
    # Run evaluation games
    logger.info("\n[3/3] Running evaluation games...")
    logger.info("-" * 40)
    
    # Dispatch all game tasks
    for _ in range(config["num_games"]):
        task_queue.put("PLAY_GAME")
    
    # Collect results
    completed_games = 0
    mafia_wins = 0
    town_wins = 0
    game_results = []
    batches_processed = 0
    
    # Metrics for per-game logging
    metrics_file = output_dir / "metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["game_num", "game_id", "winner", "total_rounds", "mafia_survived", "town_survived"])
    
    try:
        while completed_games < config["num_games"]:
            # Process model requests
            batch = []
            start_wait = time.time()
            
            while len(batch) < model_server.batch_size:
                try:
                    req = request_queue.get(timeout=0.001)
                    batch.append(req)
                except queue.Empty:
                    if batch and (time.time() - start_wait > 0.01):
                        break
                    
                    # Check for completed games
                    try:
                        res = result_queue.get_nowait()
                        if res["status"] == "success":
                            completed_games += 1
                            game_result = res["game_result"]
                            game_results.append(game_result)
                            
                            winner = str(game_result['winner']).lower()
                            if winner == "mafia":
                                mafia_wins += 1
                            else:
                                town_wins += 1
                            
                            # Log progress
                            current_mafia_wr = mafia_wins / completed_games
                            logger.info(f"  Game {completed_games}/{config['num_games']}: "
                                       f"Winner={game_result['winner']}, "
                                       f"Rounds={game_result['total_rounds']}, "
                                       f"Mafia WR={current_mafia_wr:.1%}")
                            
                            # Save individual game metrics
                            game_state = game_result["game_state"]
                            mafia_survived = sum(1 for pid in game_state.alive_players 
                                               if game_state.roles[pid].is_mafia)
                            town_survived = sum(1 for pid in game_state.alive_players 
                                              if not game_state.roles[pid].is_mafia)
                            
                            with open(metrics_file, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    completed_games,
                                    game_result["game_id"],
                                    game_result["winner"],
                                    game_result["total_rounds"],
                                    mafia_survived,
                                    town_survived
                                ])
                            
                            # Save game trace
                            trace_file = traces_dir / f"game_{completed_games:03d}_{game_result['game_id'][:8]}.json"
                            serializable_result = {
                                "game_id": game_result["game_id"],
                                "winner": game_result["winner"],
                                "total_rounds": game_result["total_rounds"],
                                "cot_history": game_result["cot_history"],
                                "final_state": game_result["final_state"]
                            }
                            with open(trace_file, "w") as f:
                                json.dump(serializable_result, f, indent=2)
                            
                            # Save formatted log (every 10th game or if small eval)
                            if config["num_games"] <= 20 or completed_games % 10 == 0:
                                log_content = format_game_log(game_result, config["cot_visibility"])
                                log_file = traces_dir / f"game_{completed_games:03d}_log.txt"
                                with open(log_file, "w") as f:
                                    f.write(log_content)
                                    
                        elif res["status"] == "error":
                            logger.error(f"  Worker error: {res['error']}")
                            completed_games += 1
                            
                    except queue.Empty:
                        pass
                    
                    if len(batch) == 0 and completed_games >= config["num_games"]:
                        break
            
            if batch:
                model_server._process_batch(batch)
                batches_processed += 1
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        
        final_mafia_wr = mafia_wins / completed_games if completed_games > 0 else 0
        final_town_wr = town_wins / completed_games if completed_games > 0 else 0
        
        logger.info(f"\nResults ({completed_games} games):")
        logger.info(f"  Mafia Win Rate: {final_mafia_wr:.2%} ({mafia_wins} wins)")
        logger.info(f"  Town Win Rate:  {final_town_wr:.2%} ({town_wins} wins)")
        
        # Average game length
        avg_rounds = np.mean([r["total_rounds"] for r in game_results]) if game_results else 0
        std_rounds = np.std([r["total_rounds"] for r in game_results]) if game_results else 0
        logger.info(f"  Avg Game Length: {avg_rounds:.1f} ± {std_rounds:.1f} rounds")
        
        # Save summary
        summary = {
            "model_type": config["model_type"],
            "checkpoint": config.get("checkpoint"),
            "num_games": completed_games,
            "mafia_wins": mafia_wins,
            "town_wins": town_wins,
            "mafia_win_rate": final_mafia_wr,
            "town_win_rate": final_town_wr,
            "avg_game_length": avg_rounds,
            "std_game_length": std_rounds,
            "config": config_save
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        
    finally:
        # Cleanup workers
        logger.info("\nStopping workers...")
        for _ in workers:
            task_queue.put("STOP")
        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        logger.info("Evaluation finished!")


def main():
    """
    Main function - edit the config dict below to change evaluation settings.
    """
    
    # ==========================================================================
    # CONFIGURATION - Edit this section to change evaluation settings
    # ==========================================================================
    config = {
        # Model Configuration
        # Options: "base" (no fine-tuning), "private", "public", "custom"
        "model_type": "base",
        
        # Checkpoint path (required for non-base models, set to None for base)
        # Examples:
        #   - "checkpoints-private-cot-run1/checkpoint-15"
        #   - "checkpoints-public-cot-run1/checkpoint-999"
        "checkpoint": None,
        
        # Base model name
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        
        # Game Configuration
        "num_players": 6,
        "role_distribution": {
            RoleType.MAFIA: 1,
            RoleType.DOCTOR: 1,
            RoleType.VILLAGER: 4
        },
        "cot_visibility": VisibilityMode.PUBLIC,
        
        # Evaluation Settings
        "num_games": 100,               # Number of games to run
        "num_workers": 8,               # Number of parallel workers
        "use_4bit": True,               # Use 4-bit quantization
        "generation_temperature": 0.6,  # Generation temperature
        "seed": 2881,                     # Random seed
        
        # Output directory (set to None for auto-generated path)
        # Example: "eval_games/baseline_run1"
        "output_dir": "eval_games/base_eval",
    }
    # ==========================================================================
    
    # Validation
    if config["model_type"] != "base" and config.get("checkpoint") is None:
        raise ValueError("checkpoint is required for non-base models")
    
    run_evaluation(config)


if __name__ == "__main__":
    main()

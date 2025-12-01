import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

# Add project root to path
sys.path.append(os.getcwd())

from mafia_experiment.training import ModelManager
from mafia_experiment.game import GameEngine
from mafia_experiment.agents import LLMAgent
from mafia_experiment.game.roles import RoleType
from mafia_experiment.cot import CoTManager, VisibilityMode

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_game_log(game_result: Dict[str, Any], cot_visibility: VisibilityMode) -> str:
    """
    Format game result into a readable story-like log
    """
    game_id = game_result["game_id"]
    winner = game_result["winner"]
    total_rounds = game_result["total_rounds"]
    game_state = game_result["game_state"]
    cot_history = game_result.get("cot_history", [])
    
    # Sort CoT history by timestamp/round/phase
    # Assuming cot_history is a list of dicts from CoTManager
    
    log = []
    log.append(f"==================================================")
    log.append(f"GAME LOG: {game_id}")
    log.append(f"WINNER: {winner.upper()}")
    log.append(f"TOTAL ROUNDS: {total_rounds}")
    log.append(f"COT VISIBILITY: {cot_visibility.value.upper()}")
    log.append(f"==================================================\n")
    
    # Role Reveal
    log.append("--- ROLE ASSIGNMENTS ---")
    for agent_id, role in game_state.roles.items():
        log.append(f"{agent_id}: {role.role_type.value} ({role.team})")
    log.append("\n")
    
    # Process by Round
    current_round = 0
    
    # Group CoTs by round and phase
    cots_by_round = {}
    for entry in cot_history:
        r = entry["round_number"]
        if r not in cots_by_round:
            cots_by_round[r] = []
        cots_by_round[r].append(entry)
        
    # Get public events from game state
    public_events = game_state.public_events
    
    for r in range(1, total_rounds + 1):
        log.append(f"=== ROUND {r} ===\n")
        
        round_cots = cots_by_round.get(r, [])
        
        # --- NIGHT PHASE ---
        log.append(f"--- NIGHT PHASE (Round {r}) ---")
        night_cots = [c for c in round_cots if c["phase"] == "night"]
        
        for cot in night_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            
            log.append(f"\n[Agent {agent_id} ({role_str}) - Night Action]")
            log.append(f"Action Type: {cot['action_type']}")
            log.append("RAW OUTPUT:")
            log.append(f"{cot['cot_text']}")
            
        # Find night events (kills)
        # We can look at public events that start with "Round {r}:" and contain "killed" or "eliminated"
        # But public events are just strings.
        # Better to look at phase history if available, but game_result might not have full phase objects easily accessible
        # Let's use public events for outcomes
        
        log.append("\n--- NIGHT OUTCOME ---")
        round_events = [e for e in public_events if e.startswith(f"Round {r}:")]
        for event in round_events:
            if "killed at night" in event:
                log.append(f"EVENT: {event}")
        
        # --- DAY PHASE ---
        log.append(f"\n--- DAY PHASE (Round {r}) ---")
        
        # Discussion 1
        log.append(f"\n--- DISCUSSION ROUND 1 ---")
        discuss_1_cots = [c for c in round_cots if c["phase"] == "day" and c["action_type"] == "discuss_1"]
        for cot in discuss_1_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            
            log.append(f"\n[Agent {agent_id} ({role_str})]")
            log.append("RAW OUTPUT:")
            log.append(f"{cot['cot_text']}")

        # Discussion 2
        log.append(f"\n--- DISCUSSION ROUND 2 ---")
        discuss_2_cots = [c for c in round_cots if c["phase"] == "day" and c["action_type"] == "discuss_2"]
        for cot in discuss_2_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            
            log.append(f"\n[Agent {agent_id} ({role_str})]")
            log.append("RAW OUTPUT:")
            log.append(f"{cot['cot_text']}")

        # Voting
        log.append(f"\n--- VOTING ---")
        vote_cots = [c for c in round_cots if c["phase"] == "day" and c["action_type"] == "vote"]
        for cot in vote_cots:
            agent_id = cot["agent_id"]
            role = game_state.roles.get(agent_id)
            role_str = role.role_type.value if role else "Unknown"
            
            log.append(f"\n[Agent {agent_id} ({role_str})]")
            log.append("RAW OUTPUT:")
            log.append(f"{cot['cot_text']}")
            # Extract vote target from CoT or action if possible. 
            # The CoT text usually ends with "ACTION: Vote <Target>"
            
        log.append("\n--- DAY OUTCOME ---")
        for event in round_events:
            if "eliminated" in event and "killed at night" not in event:
                log.append(f"EVENT: {event}")
                
        log.append("\n")

    log.append(f"GAME OVER. Winner: {winner}")
    return "\n".join(log)

def main():
    # --- CONFIGURATION ---
    CHECKPOINT_PATH = None  # Set to None to use base model, or path like "checkpoints/checkpoint-10"
    NUM_GAMES = 3
    OUTPUT_DIR = "logs/evaluation_stories"
    COT_VISIBILITY = VisibilityMode.PUBLIC # or VisibilityMode.PRIVATE
    
    # Game Config
    NUM_PLAYERS = 6
    ROLE_DISTRIBUTION = {
        RoleType.MAFIA: 1,
        RoleType.DOCTOR: 1,
        RoleType.VILLAGER: 4
    }
    
    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Evaluation Run")
    if CHECKPOINT_PATH:
        logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    else:
        logger.info(f"Using Base Model (No Checkpoint)")
        
    logger.info(f"Num Games: {NUM_GAMES}")
    logger.info(f"CoT Visibility: {COT_VISIBILITY.value}")
    
    # 1. Load Model
    if CHECKPOINT_PATH and not os.path.exists(CHECKPOINT_PATH):
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    logger.info("Loading model...")
    model_manager = ModelManager(
        model_name="meta-llama/Llama-3.2-1B-Instruct",  # Instruct-tuned base model
        use_4bit=True
    )
    
    if CHECKPOINT_PATH:
        # Load checkpoint
        model, tokenizer = model_manager.load_checkpoint(CHECKPOINT_PATH)
    else:
        # Load base model with LoRA (untrained) or just base model
        # Let's stick to load_model_with_lora() for consistency in architecture, 
        # effectively evaluating "Randomly Initialized LoRA" which should be close to base model 
        # (since LoRA initializes to identity/zero usually).
        model, tokenizer = model_manager.load_model_with_lora()
    # model, tokenizer = model_manager.load_checkpoint(CHECKPOINT_PATH)
    
    # 2. Run Games
    for i in range(1, NUM_GAMES + 1):
        logger.info(f"Playing Game {i}/{NUM_GAMES}...")
        
        # Setup Agents
        players = []
        for j in range(NUM_PLAYERS):
            agent = LLMAgent(
                agent_id=f"Player_{j}",
                role=None, # Assigned by game engine
                model=model,
                tokenizer=tokenizer,
                temperature=0.3
            )
            players.append(agent)
            
        # Setup CoT Manager
        cot_manager = CoTManager(visibility_mode=COT_VISIBILITY)
        
        # Setup Game Engine
        engine = GameEngine(
            players=players,
            role_distribution=ROLE_DISTRIBUTION,
            collect_trajectories=False, # No training needed
            cot_manager=cot_manager,
            model=model,
            tokenizer=tokenizer
        )
        
        # Run Game
        try:
            game_result = engine.run_game(max_rounds=15)
            
            # DEBUG: Check CoT history
            cot_history = game_result.get("cot_history", [])
            print(f"DEBUG: Number of CoT entries: {len(cot_history)}")
            if cot_history:
                print(f"DEBUG: Sample CoT entry keys: {cot_history[0].keys()}")
                print(f"DEBUG: Sample CoT entry: {cot_history[0]}")
            else:
                print("DEBUG: cot_history is EMPTY!")
            
            # Format Log
            log_content = format_game_log(game_result, COT_VISIBILITY)
            
            # Save Log
            filename = f"game_{i}_winner_{game_result['winner']}.txt"
            filepath = Path(OUTPUT_DIR) / filename
            with open(filepath, "w") as f:
                f.write(log_content)
                
            logger.info(f"  Game {i} finished. Winner: {game_result['winner']}. Log saved to {filepath}")
            
        except Exception as e:
            logger.error(f"  Game {i} failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Evaluation Complete!")

if __name__ == "__main__":
    main()

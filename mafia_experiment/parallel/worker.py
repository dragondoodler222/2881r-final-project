import torch
import numpy as np
import random
import os
from typing import Dict, Any, List
from ..game.game_engine import GameEngine
from ..agents.llm_agent import LLMAgent
from ..cot.cot_manager import CoTManager, VisibilityMode
from ..game.roles import RoleType
from .remote_client import RemoteModelClient


# Redefining to include task_queue
def worker_process(
    worker_id: int,
    request_queue,
    response_queue,
    task_queue,
    result_queue,
    config: Dict[str, Any]
):
    # Disable CUDA for workers to save memory
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Seeding
    seed = config.get('seed', 42) + (worker_id * 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    client = RemoteModelClient(request_queue, response_queue, worker_id)
    
    while True:
        task = task_queue.get()
            
        if task == "STOP":
            break
            
        if task == "PLAY_GAME":
            # Setup game
            agents = []
            for i in range(config["num_players"]):
                agent_id = f"player_{i+1}"
                # We pass None for model/tokenizer since we use client
                agent = LLMAgent(
                    agent_id=agent_id,
                    role=None,
                    model=None, 
                    tokenizer=None,
                    temperature=config.get("generation_temperature", 0.7)
                )
                agents.append(agent)
                
            cot_manager = CoTManager(visibility_mode=config["cot_visibility"])
            
            # Initialize engine with client
            # We need to patch GameEngine to accept client or use duck typing
            # We will pass client as 'model' and None as 'tokenizer'
            # GameEngine._batch_generate will check for generate_batch method
            
            game_engine = GameEngine(
                players=agents,
                role_distribution=config["role_distribution"],
                collect_trajectories=True,
                cot_manager=cot_manager,
                model=client, # Pass client as model
                tokenizer=None # Tokenizer handled on server
            )
            
            # Run game
            try:
                game_result = game_engine.run_game(max_rounds=10)
                
                # Send result back
                # We need to serialize what's needed.
                # Trajectories are needed. Game stats are needed.
                # GameState object might be heavy/not picklable? It should be fine.
                
                result_queue.put({
                    "status": "success",
                    "worker_id": worker_id,
                    "game_result": game_result
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                result_queue.put({
                    "status": "error",
                    "worker_id": worker_id,
                    "error": str(e)
                })

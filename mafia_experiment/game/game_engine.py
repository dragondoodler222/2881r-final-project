"""
Game Engine: Core Mafia game loop and rules enforcement
"""

import uuid
import torch
from typing import List, Dict, Optional, Any
from .game_state import GameState
from .roles import Role, RoleType, create_role
from .phases import Phase, PhaseType, NightPhaseResult, DayPhaseResult
from ..agents.base_agent import BaseAgent, AgentAction
from ..agents.llm_agent import LLMAgent
from ..training.trajectory_buffer import Trajectory
from ..cot.cot_manager import CoTManager, VisibilityMode


class GameEngine:
    """
    Manages the Mafia game flow and rule enforcement
    """

    def __init__(
        self,
        players: List[BaseAgent],
        role_distribution: Optional[Dict[RoleType, int]] = None,
        collect_trajectories: bool = False,
        cot_manager: Optional[CoTManager] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize game engine

        Args:
            players: List of agent objects
            role_distribution: Number of each role (if None, uses default)
            collect_trajectories: Whether to collect trajectories for RL training
            cot_manager: Chain of Thought manager for visibility control
            model: The LLM model (for batch generation)
            tokenizer: The tokenizer (for batch generation)
        """
        self.game_id = str(uuid.uuid4())
        self.players = players
        self.game_state: Optional[GameState] = None
        self.current_phase: Optional[Phase] = None
        self.collect_trajectories = collect_trajectories
        self.trajectories: List = []  # Will store Trajectory objects if collect_trajectories=True
        self.cot_manager = cot_manager
        self.model = model
        self.tokenizer = tokenizer

        # Default role distribution: 2 Mafia, 1 Doctor, rest Villagers
        if role_distribution is None:
            num_players = len(players)
            role_distribution = {
                RoleType.MAFIA: 2,
                RoleType.DOCTOR: 1,
                RoleType.VILLAGER: num_players - 3
            }

        self.role_distribution = role_distribution

    def _batch_generate(self, prompts: List[str], max_tokens=256, temperature=0.7) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of prompts
        """
        # Check if model is a RemoteModelClient (duck typing)
        if hasattr(self.model, 'generate_batch'):
            return self.model.generate_batch(prompts, max_tokens, temperature)

        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be provided for batch generation")

        # Ensure left padding for generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        # Tokenize
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Restore padding side
        self.tokenizer.padding_side = original_padding_side
            
        # Process outputs for each prompt
        results = []
        for i in range(len(prompts)):
            # Extract sequence (remove prompt)
            prompt_len = inputs.input_ids.shape[1]
            gen_ids = outputs.sequences[i][prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # Construct scores for this sample
            # outputs.scores is a tuple of len(generated_tokens). Each element is (batch_size, vocab_size).
            # We need to slice it to get (1, vocab_size) for each step for this sample
            sample_scores = tuple(score[i:i+1] for score in outputs.scores)
            
            # Compute log prob using LLMAgent's static method
            log_prob = LLMAgent.compute_log_prob_from_scores(sample_scores, gen_ids)
            
            results.append({
                "text": text,
                "log_prob": log_prob,
                "input_ids": inputs.input_ids[i].cpu(),
                "generated_ids": gen_ids.cpu()
            })
        return results

    def _run_batch_action(
        self, 
        agents_to_act: List[BaseAgent], 
        action_type_map: Dict[str, str], 
        phase_name: str,
        is_cot_public: bool = False
    ) -> Dict[str, AgentAction]:
        """
        Run a batch of actions for multiple agents
        
        Args:
            agents_to_act: List of agents
            action_type_map: Map of agent_id -> action_type
            phase_name: Name of phase for trajectory logging
            is_cot_public: Whether CoT is public
            
        Returns:
            Map of agent_id -> AgentAction
        """
        if not agents_to_act:
            return {}

        # 1. Prepare prompts
        prompts = []
        valid_agents = []
        
        for agent in agents_to_act:
            if not isinstance(agent, LLMAgent):
                continue
                
            visible_state = self.game_state.get_visible_state(agent.agent_id)
            visible_cots = []
            if self.cot_manager:
                visible_cots = self.cot_manager.get_visible_cots(
                    agent.agent_id, 
                    self.game_state.current_round
                )
            
            action_type = action_type_map[agent.agent_id]
            prompt = agent.prepare_act(action_type, visible_state, visible_cots, is_cot_public)
            prompts.append(prompt)
            valid_agents.append(agent)
            
        if not prompts:
            return {}
            
        # 2. Batch Generate
        # Optimize max_tokens based on action type
        # If all actions are "vote" or "kill" or "save", we can use fewer tokens
        # Discussion needs more tokens
        
        current_action_type = list(action_type_map.values())[0] if action_type_map else "unknown"
        
        if "discuss" in current_action_type:
            max_tokens = 256
        else:
            max_tokens = 64 # Voting/Actions are short
            
        results = self._batch_generate(prompts, max_tokens=max_tokens)
        
        # 3. Finalize actions
        actions = {}
        for i, agent in enumerate(valid_agents):
            res = results[i]
            action_type = action_type_map[agent.agent_id]
            
            action = agent.finalize_act(
                action_type=action_type,
                prompt=prompts[i],
                generated_text=res["text"],
                log_prob=res["log_prob"],
                input_ids=res["input_ids"],
                generated_ids=res["generated_ids"],
                game_state=self.game_state.get_visible_state(agent.agent_id)
            )
            
            actions[agent.agent_id] = action
            
            # Collect trajectory immediately? Or let caller handle it?
            # Caller might need to do extra processing (like extracting public arg)
            # But we can collect basic trajectory here
            # Note: visible_cots needs to be re-fetched or passed through if we want to log it accurately
            # For simplicity, we re-fetch (it's cheap)
            visible_cots = []
            if self.cot_manager:
                visible_cots = self.cot_manager.get_visible_cots(
                    agent.agent_id, 
                    self.game_state.current_round
                )
            self._collect_trajectory(agent, action, phase_name, visible_cots)
            
        return actions

    def initialize_game(self) -> GameState:
        """
        Initialize the game state and assign roles

        Returns:
            Initial game state
        """
        import random

        # Reset all agents
        for agent in self.players:
            agent.reset()

        # Create role assignments
        roles_to_assign = []
        for role_type, count in self.role_distribution.items():
            for _ in range(count):
                roles_to_assign.append(create_role(role_type))

        # Shuffle and assign roles
        random.shuffle(roles_to_assign)
        role_assignments = {}
        for i, agent in enumerate(self.players):
            role = roles_to_assign[i]
            role_assignments[agent.agent_id] = role
            agent.role = role

        # Create game state
        player_ids = [agent.agent_id for agent in self.players]
        self.game_state = GameState(
            game_id=self.game_id,
            players=player_ids,
            roles=role_assignments
        )

        # Update CoT manager with roles
        if self.cot_manager:
            self.cot_manager.set_role_assignments(role_assignments)

        return self.game_state

    def run_game(self, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Run a complete game until win condition is met

        Args:
            max_rounds: Maximum number of rounds before draw

        Returns:
            Game result dictionary
        """
        # Always initialize a new game to ensure random role assignment and fresh state
        self.initialize_game()

        self.game_state.current_round = 0
        self.trajectories = []  # Reset trajectories for new game

        while self.game_state.current_round < max_rounds:
            self.game_state.current_round += 1

            # Night phase
            night_result = self.run_night_phase()

            # Check win condition
            winner = self.game_state.check_win_condition()
            if winner:
                return self._create_game_result(winner)

            # Day phase
            day_result = self.run_day_phase()

            # Check win condition
            winner = self.game_state.check_win_condition()
            if winner:
                return self._create_game_result(winner)

        # Max rounds reached - determine winner by team size
        alive_mafia = len(self.game_state.get_alive_mafia())
        alive_village = len(self.game_state.get_alive_village())

        winner = "mafia" if alive_mafia >= alive_village else "village"
        return self._create_game_result(winner, draw=True)

    def _collect_trajectory(
        self,
        agent: BaseAgent,
        action: AgentAction,
        phase_type: str,
        visible_cots: List[Dict]
    ) -> None:
        """Helper to collect trajectory from an agent action"""
        if not self.collect_trajectories:
            return

        # Check if agent has necessary attributes (is LLMAgent)
        if not hasattr(agent, 'get_last_log_prob'):
            return

        # Get data from agent
        log_prob = agent.get_last_log_prob()
        prompt = agent.get_last_prompt()
        input_ids = agent.get_last_input_ids()
        generated_ids = agent.get_last_generated_ids()
        cot = agent.get_last_cot()
        parsing_confidence = getattr(agent, 'get_last_parsing_confidence', lambda: 1.0)()

        if log_prob is None:
            return

        # Create trajectory
        traj = Trajectory(
            agent_id=agent.agent_id,
            game_id=self.game_id,
            round_number=self.game_state.current_round,
            phase=phase_type,
            state=self.game_state.get_visible_state(agent.agent_id),
            action=str(action.target) if action.target else action.action_type,
            log_prob=log_prob,
            reward=0.0,  # Will be assigned later
            cot=cot,
            visible_cots=visible_cots,
            prompt=prompt,
            input_ids=input_ids,
            generated_ids=generated_ids,
            parsing_confidence=parsing_confidence
        )
        
        self.trajectories.append(traj)

    def run_night_phase(self) -> NightPhaseResult:
        """
        Execute the night phase

        Returns:
            Night phase result
        """
        self.current_phase = Phase(PhaseType.NIGHT, self.game_state.current_round)
        self.game_state.current_phase = PhaseType.NIGHT

        actions = []
        mafia_target = None
        doctor_save = None

        # Collect agents who can act
        agents_to_act = []
        action_type_map = {}
        
        for agent in self.players:
            if not self.game_state.is_alive(agent.agent_id) or not agent.has_night_action():
                continue
            
            if agent.role.role_type == RoleType.MAFIA:
                agents_to_act.append(agent)
                action_type_map[agent.agent_id] = "kill"
            elif agent.role.role_type == RoleType.DOCTOR:
                agents_to_act.append(agent)
                action_type_map[agent.agent_id] = "save"
        
        # Run batch action
        batch_actions = self._run_batch_action(agents_to_act, action_type_map, "night")
        
        # Process actions
        for agent_id, action in batch_actions.items():
            agent = next(a for a in self.players if a.agent_id == agent_id)
            
            # Record CoT
            if self.cot_manager:
                self.cot_manager.record_cot(
                    agent_id=agent.agent_id,
                    cot_text=getattr(agent, 'last_cot', ''),
                    round_number=self.game_state.current_round,
                    phase="night",
                    action_type=action.action_type
                )
            
            actions.append(action)
            
            if action.action_type == "kill":
                if mafia_target is None and action.target:
                    mafia_target = action.target
            elif action.action_type == "save":
                if action.target:
                    doctor_save = action.target

        # Resolve night actions
        killed_player = None
        # Only kill if: target exists, target is alive, and not saved by doctor
        if mafia_target and mafia_target in self.game_state.alive_players and mafia_target != doctor_save:
            killed_player = mafia_target
            self.game_state.kill_player(killed_player, "killed at night")

        result = NightPhaseResult(
            phase=self.current_phase,
            actions=[vars(a) for a in actions],  # Store action dicts
            mafia_target=mafia_target,
            doctor_save=doctor_save,
            killed_player=killed_player
        )

        self.game_state.phase_history.append(result)
        return result

    def run_day_phase(self) -> DayPhaseResult:
        """
        Execute the day phase with structured discussion and voting

        Structure:
        1. Discussion Round 1: Each agent contributes 1 argument
        2. Read phase: All agents see all round 1 arguments
        3. Discussion Round 2: Each agent contributes 1 more argument
        4. Voting: Each agent votes
        5. Resolution: Exactly one player eliminated (ties broken randomly)

        Returns:
            Day phase result
        """
        self.current_phase = Phase(PhaseType.DAY, self.game_state.current_round)
        self.game_state.current_phase = PhaseType.DAY

        # Use GameState to determine alive agents
        alive_agents = [agent for agent in self.players if self.game_state.is_alive(agent.agent_id)]

        # Determine if CoT is public
        is_cot_public = False
        if self.cot_manager and self.cot_manager.visibility_mode == VisibilityMode.PUBLIC:
            is_cot_public = True

        # === DISCUSSION ROUND 1 ===
        round_1_arguments = []
        round_1_cots_to_add = []
        
        # Batch Discussion 1
        action_type_map = {agent.agent_id: "discuss_1" for agent in alive_agents}
        batch_actions_1 = self._run_batch_action(alive_agents, action_type_map, "day_discuss_1", is_cot_public)
        
        # Process results
        # We need to maintain order of alive_agents for consistency if needed, but dict iteration is fine
        for agent in alive_agents:
            if agent.agent_id not in batch_actions_1:
                continue
                
            action = batch_actions_1[agent.agent_id]
            cot_text = getattr(agent, 'last_cot', '')
            
            # Extract public argument
            import re
            public_arg = ""
            public_match = re.search(r'PUBLIC ARGUMENT:(.*?)ACTION:', cot_text + "ACTION:", re.DOTALL | re.IGNORECASE)
            if public_match:
                public_arg = public_match.group(1).strip()
            
            round_1_cots_to_add.append({
                "agent_id": agent.agent_id,
                "cot_text": cot_text,
                "public_argument": public_arg,
                "action_type": "discuss_1"
            })
            
            round_1_arguments.append({
                "agent_id": agent.agent_id,
                "argument": public_arg if public_arg else action.reasoning,
                "cot": cot_text
            })

        # Record all CoTs from Round 1 (Simultaneous revelation)
        if self.cot_manager:
            for item in round_1_cots_to_add:
                self.cot_manager.record_cot(
                    agent_id=item["agent_id"],
                    cot_text=item["cot_text"],
                    round_number=self.game_state.current_round,
                    phase="day",
                    action_type=item["action_type"],
                    public_argument=item.get("public_argument", "")
                )

        # === READ PHASE ===
        # All agents now have access to round 1 arguments via CoTManager

        # === DISCUSSION ROUND 2 ===
        round_2_arguments = []
        round_2_cots_to_add = []

        # Batch Discussion 2
        action_type_map = {agent.agent_id: "discuss_2" for agent in alive_agents}
        batch_actions_2 = self._run_batch_action(alive_agents, action_type_map, "day_discuss_2", is_cot_public)

        for agent in alive_agents:
            if agent.agent_id not in batch_actions_2:
                continue
                
            action = batch_actions_2[agent.agent_id]
            cot_text = getattr(agent, 'last_cot', '')
            
            # Extract public argument
            import re
            public_arg = ""
            public_match = re.search(r'PUBLIC ARGUMENT:(.*?)ACTION:', cot_text + "ACTION:", re.DOTALL | re.IGNORECASE)
            if public_match:
                public_arg = public_match.group(1).strip()
            
            round_2_cots_to_add.append({
                "agent_id": agent.agent_id,
                "cot_text": cot_text,
                "public_argument": public_arg,
                "action_type": "discuss_2"
            })
            
            round_2_arguments.append({
                "agent_id": agent.agent_id,
                "argument": public_arg if public_arg else action.reasoning,
                "cot": cot_text
            })

        # Record all CoTs from Round 2
        if self.cot_manager:
            for item in round_2_cots_to_add:
                self.cot_manager.record_cot(
                    agent_id=item["agent_id"],
                    cot_text=item["cot_text"],
                    round_number=self.game_state.current_round,
                    phase="day",
                    action_type=item["action_type"],
                    public_argument=item.get("public_argument", "")
                )

        # === VOTING ===
        votes = {}
        all_arguments = round_1_arguments + round_2_arguments

        # Batch Voting
        action_type_map = {agent.agent_id: "vote" for agent in alive_agents}
        batch_actions_vote = self._run_batch_action(alive_agents, action_type_map, "day_vote", is_cot_public)

        for agent in alive_agents:
            if agent.agent_id not in batch_actions_vote:
                continue
                
            action = batch_actions_vote[agent.agent_id]
            
            # Record CoT for vote
            if self.cot_manager:
                self.cot_manager.record_cot(
                    agent_id=agent.agent_id,
                    cot_text=getattr(agent, 'last_cot', ''),
                    round_number=self.game_state.current_round,
                    phase="day",
                    action_type="vote"
                )
            
            if action.target and action.target in self.game_state.alive_players:
                votes[agent.agent_id] = action.target

        # === VOTE RESOLUTION ===

        # Count votes
        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # ENFORCE: Exactly one player must be eliminated
        if vote_counts:
            # Find player(s) with most votes
            max_votes = max(vote_counts.values())
            candidates = [p for p, v in vote_counts.items() if v == max_votes]

            # Break ties randomly
            import random
            lynched_player = random.choice(candidates)
        else:
            # No votes cast - randomly eliminate someone
            import random
            lynched_player = random.choice(list(self.game_state.alive_players))

        # Execute elimination
        self.game_state.kill_player(lynched_player, "lynched")

        # Collect all actions
        day_actions = []
        if 'batch_actions_1' in locals():
            day_actions.extend([vars(a) for a in batch_actions_1.values()])
        if 'batch_actions_2' in locals():
            day_actions.extend([vars(a) for a in batch_actions_2.values()])
        if 'batch_actions_vote' in locals():
            day_actions.extend([vars(a) for a in batch_actions_vote.values()])

        result = DayPhaseResult(
            phase=self.current_phase,
            actions=day_actions,
            discussion_round_1=round_1_arguments,
            discussion_round_2=round_2_arguments,
            votes=votes,
            vote_counts=vote_counts,
            lynched_player=lynched_player
        )

        self.game_state.phase_history.append(result)
        return result

    def _create_game_result(
        self,
        winner: str,
        draw: bool = False
    ) -> Dict[str, Any]:
        """Create final game result dictionary"""
        cot_history = []
        if self.cot_manager:
            cot_history = [entry.to_dict() for entry in self.cot_manager.cot_log]

        return {
            "game_id": self.game_id,
            "winner": winner,
            "total_rounds": self.game_state.current_round,
            "game_state": self.game_state,
            "draw": draw,
            "final_state": self.game_state.to_dict(),
            "trajectories": self.trajectories,
            "cot_history": cot_history
        }

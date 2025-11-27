"""
Game Engine: Core Mafia game loop and rules enforcement
"""

import uuid
from typing import List, Dict, Optional, Any
from .game_state import GameState
from .roles import Role, RoleType, create_role
from .phases import Phase, PhaseType, NightPhaseResult, DayPhaseResult
from ..agents.base_agent import BaseAgent, AgentAction
from ..training.trajectory_buffer import Trajectory


class GameEngine:
    """
    Manages the Mafia game flow and rule enforcement
    """

    def __init__(
        self,
        players: List[BaseAgent],
        role_distribution: Optional[Dict[RoleType, int]] = None,
        collect_trajectories: bool = False
    ):
        """
        Initialize game engine

        Args:
            players: List of agent objects
            role_distribution: Number of each role (if None, uses default)
            collect_trajectories: Whether to collect trajectories for RL training
        """
        self.game_id = str(uuid.uuid4())
        self.players = players
        self.game_state: Optional[GameState] = None
        self.current_phase: Optional[Phase] = None
        self.collect_trajectories = collect_trajectories
        self.trajectories: List = []  # Will store Trajectory objects if collect_trajectories=True

        # Default role distribution: 2 Mafia, 1 Doctor, rest Villagers
        if role_distribution is None:
            num_players = len(players)
            role_distribution = {
                RoleType.MAFIA: 2,
                RoleType.DOCTOR: 1,
                RoleType.VILLAGER: num_players - 3
            }

        self.role_distribution = role_distribution

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

        # Collect actions from agents with night actions
        for agent in self.players:
            # Use GameState to check if agent is alive, not the agent object itself
            if not self.game_state.is_alive(agent.agent_id) or not agent.has_night_action():
                continue

            # Get visible state and CoTs for this agent
            visible_state = self.game_state.get_visible_state(agent.agent_id)
            visible_cots = []  # Will be populated by CoT manager in full implementation

            # Determine action type based on role
            if agent.role.role_type == RoleType.MAFIA:
                action = agent.act("kill", visible_state, visible_cots)
                self._collect_trajectory(agent, action, "night", visible_cots)
                actions.append(action)
                # Mafia vote on target (simplified: take first Mafia's choice)
                if mafia_target is None and action.target:
                    mafia_target = action.target

            elif agent.role.role_type == RoleType.DOCTOR:
                action = agent.act("save", visible_state, visible_cots)
                self._collect_trajectory(agent, action, "night", visible_cots)
                actions.append(action)
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
            actions=[],  # Store action dicts
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

        # === DISCUSSION ROUND 1 ===
        round_1_arguments = []

        for agent in alive_agents:
            visible_state = self.game_state.get_visible_state(agent.agent_id)
            visible_cots = []  # Will be populated by CoT manager

            # Agent contributes first argument
            action = agent.act("discuss_1", visible_state, visible_cots)
            self._collect_trajectory(agent, action, "day_discuss_1", visible_cots)
            
            round_1_arguments.append({
                "agent_id": agent.agent_id,
                "argument": action.reasoning,
                "cot": getattr(agent, 'last_cot', '')
            })

        # === READ PHASE ===
        # All agents now have access to round 1 arguments
        # (Will be passed as visible_cots in round 2)

        # === DISCUSSION ROUND 2 ===
        round_2_arguments = []

        for agent in alive_agents:
            visible_state = self.game_state.get_visible_state(agent.agent_id)
            # Include round 1 arguments as visible CoTs
            visible_cots = round_1_arguments

            # Agent contributes second argument
            action = agent.act("discuss_2", visible_state, visible_cots)
            self._collect_trajectory(agent, action, "day_discuss_2", visible_cots)
            
            round_2_arguments.append({
                "agent_id": agent.agent_id,
                "argument": action.reasoning,
                "cot": getattr(agent, 'last_cot', '')
            })

        # === VOTING ===
        votes = {}
        all_arguments = round_1_arguments + round_2_arguments

        for agent in alive_agents:
            visible_state = self.game_state.get_visible_state(agent.agent_id)
            # Include all arguments in voting context
            visible_cots = all_arguments

            action = agent.act("vote", visible_state, visible_cots)
            self._collect_trajectory(agent, action, "day_vote", visible_cots)

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

        result = DayPhaseResult(
            phase=self.current_phase,
            actions=[],
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
        return {
            "game_id": self.game_id,
            "winner": winner,
            "total_rounds": self.game_state.current_round,
            "game_state": self.game_state,
            "draw": draw,
            "final_state": self.game_state.to_dict(),
            "trajectories": self.trajectories
        }

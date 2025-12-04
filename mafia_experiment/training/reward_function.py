"""
Reward Function: Compute rewards from game outcomes
"""

from typing import Dict, Any, List
from ..game.game_state import GameState
from ..game.phases import PhaseType, NightPhaseResult, DayPhaseResult
from ..game.roles import RoleType


class RewardFunction:
    """
    Computes rewards for agents based on game outcomes and actions
    """

    def __init__(
        self,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        survival_weight: float = 0.3,
        mafia_stealth_bonus: float = 0.2,
        # Shaping rewards (scaled down)
        mafia_kill_reward: float = 0.1,
        mafia_misvote_reward: float = 0.2,
        villager_catch_reward: float = 0.2,
        doctor_save_reward: float = 0.2
    ):
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.survival_weight = survival_weight
        self.mafia_stealth_bonus = mafia_stealth_bonus

        # Shaping rewards
        self.mafia_kill_reward = mafia_kill_reward
        self.mafia_misvote_reward = mafia_misvote_reward
        self.villager_catch_reward = villager_catch_reward
        self.doctor_save_reward = doctor_save_reward

    def compute_game_reward(
        self,
        agent_id: str,
        game_state: GameState,
        winner: str,
        rounds_survived: int,
        total_rounds: int,
        was_suspected: bool = False
    ) -> float:
        """
        Compute final reward for an agent based on game outcome

        Args:
            agent_id: Agent identifier
            game_state: Final game state
            winner: "mafia" or "village"
            rounds_survived: Number of rounds agent survived
            total_rounds: Total rounds in the game
            was_suspected: Whether the agent was heavily suspected

        Returns:
            Reward value
        """
        agent_role = game_state.roles.get(agent_id)
        if not agent_role:
            return 0.0

        # Base reward: did agent's team win?
        if (agent_role.is_mafia and winner == "mafia") or \
           (agent_role.is_village and winner == "village"):
            base_reward = self.win_reward
        else:
            base_reward = self.loss_reward

        # Survival bonus: reward lasting longer in the game
        survival_ratio = rounds_survived / max(total_rounds, 1)
        survival_bonus = self.survival_weight * survival_ratio

        # Stealth bonus for Mafia: reward not being suspected
        stealth_bonus = 0.0
        if agent_role.is_mafia and not was_suspected and base_reward > 0:
            stealth_bonus = self.mafia_stealth_bonus

        total_reward = base_reward + survival_bonus + stealth_bonus

        return total_reward

    def compute_step_reward(
        self,
        agent_id: str,
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        game_state: GameState
    ) -> float:
        """
        Compute immediate reward for a single action (optional)

        This can be used for more fine-grained reward shaping, but
        may make learning more complex. Start with just terminal rewards.

        Args:
            agent_id: Agent identifier
            action: Action taken
            outcome: Result of the action
            game_state: Current game state

        Returns:
            Step reward (default 0 for pure REINFORCE)
        """
        # For initial implementation, we use only terminal rewards
        # Can add step rewards later if needed
        return 0.0

    def assign_rewards_to_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        game_state: GameState,
        winner: str,
        total_rounds: int
    ) -> List[Dict[str, Any]]:
        """
        Assign rewards to all steps in trajectories

        Args:
            trajectories: List of trajectory dictionaries
            game_state: Final game state
            winner: Winner of the game
            total_rounds: Total rounds played

        Returns:
            Trajectories with rewards assigned
        """
        # Group trajectories by agent
        agent_trajectories = {}
        for traj in trajectories:
            agent_id = traj["agent_id"]
            if agent_id not in agent_trajectories:
                agent_trajectories[agent_id] = []
            agent_trajectories[agent_id].append(traj)

        # Compute terminal reward for each agent
        for agent_id, agent_trajs in agent_trajectories.items():
            # Use the max round_number this agent appears in as "rounds survived"
            # len(agent_trajs) would count actions (multiple per round), inflating the bonus
            rounds_survived = max(traj["round_number"] for traj in agent_trajs)

            # Calculate if agent was suspected (simplified version)
            # In full implementation, track votes against the agent
            was_suspected = False

            terminal_reward = self.compute_game_reward(
                agent_id=agent_id,
                game_state=game_state,
                winner=winner,
                rounds_survived=rounds_survived,
                total_rounds=total_rounds,
                was_suspected=was_suspected
            )

            # Determine if agent's team won (for conditional shaping)
            agent_role = game_state.roles.get(agent_id)
            team_won = False
            if agent_role:
                if (agent_role.is_mafia and winner == "mafia") or \
                   (agent_role.is_village and winner == "village"):
                    team_won = True

            # Assign rewards to trajectory steps
            for i, traj in enumerate(agent_trajs):
                step_reward = 0.0
                
                # Apply shaping rewards ONLY if team won
                if team_won and agent_role:
                    round_num = traj["round_number"]
                    phase_type = traj["phase"] # "night", "day_discuss_1", "day_vote", etc.
                    
                    # Find corresponding phase result
                    target_phase_type = None
                    if phase_type == "night":
                        target_phase_type = PhaseType.NIGHT
                    elif phase_type.startswith("day"):
                        target_phase_type = PhaseType.DAY
                        
                    phase_result = None
                    if target_phase_type:
                        for pr in game_state.phase_history:
                            if pr.phase.round_number == round_num and pr.phase.phase_type == target_phase_type:
                                phase_result = pr
                                break
                    
                    if phase_result:
                        # Night Actions
                        if phase_type == "night" and isinstance(phase_result, NightPhaseResult):
                            # Mafia Kill Reward
                            if agent_role.is_mafia and phase_result.killed_player:
                                step_reward += self.mafia_kill_reward
                            
                            # Doctor Save Reward
                            if agent_role.role_type == RoleType.DOCTOR and phase_result.doctor_save:
                                # Check if save was successful (target was mafia target)
                                if phase_result.mafia_target and phase_result.doctor_save == phase_result.mafia_target:
                                    step_reward += self.doctor_save_reward
                        
                        # Day Actions (Voting)
                        elif phase_type == "day_vote" and isinstance(phase_result, DayPhaseResult):
                            eliminated = phase_result.eliminated_player
                            if eliminated:
                                # Check if agent voted for the eliminated player
                                voted_target = phase_result.votes.get(agent_id)
                                if voted_target == eliminated:
                                    eliminated_role = game_state.roles.get(eliminated)
                                    if eliminated_role:
                                        # Mafia Misvote Reward (Mafia voting for Villager)
                                        if agent_role.is_mafia and eliminated_role.is_village:
                                            step_reward += self.mafia_misvote_reward
                                        
                                        # Villager Catch Reward (Village/Doctor voting for Mafia)
                                        if agent_role.is_village and eliminated_role.is_mafia:
                                            step_reward += self.villager_catch_reward

                if i == len(agent_trajs) - 1:
                    # Last step gets the terminal reward + any step reward
                    traj["reward"] = terminal_reward + step_reward
                else:
                    # Intermediate steps get step reward
                    traj["reward"] = step_reward

        return trajectories

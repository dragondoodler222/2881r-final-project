"""
Reward Function: Compute rewards from game outcomes
"""

from typing import Dict, Any, List
from ..game.game_state import GameState


class RewardFunction:
    """
    Computes rewards for agents based on game outcomes and actions
    """

    def __init__(
        self,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        survival_weight: float = 0.3,
        mafia_stealth_bonus: float = 0.2
    ):
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.survival_weight = survival_weight
        self.mafia_stealth_bonus = mafia_stealth_bonus

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
            rounds_survived = len(agent_trajs)

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

            # Assign rewards to trajectory steps
            # For REINFORCE, we typically use Monte Carlo returns
            # Give terminal reward to last step, 0 to others
            for i, traj in enumerate(agent_trajs):
                if i == len(agent_trajs) - 1:
                    # Last step gets the terminal reward
                    traj["reward"] = terminal_reward
                else:
                    # Intermediate steps get 0 (or step reward if using)
                    traj["reward"] = 0.0

        return trajectories

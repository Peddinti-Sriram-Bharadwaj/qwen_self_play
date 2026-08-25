import torch
from self_play import collect_batched_self_play_trajectories
from envs.base_env import LLMEnvironment

class Evaluator:
    def __init__(self, llm_env: LLMEnvironment, storage, num_envs: int = 100):
        self.llm_env = llm_env
        self.storage = storage
        self.num_envs = num_envs
        
    def evaluate_matchup(self, learning_agent, opponent_agent, matchup_name: str) -> dict:
        """
        Pits learning_agent (Player 0) against opponent_agent (Player 1) for num_envs games.
        Returns a dictionary of metrics.
        """
        print(f"Evaluating: {matchup_name} ({self.num_envs} games)...")
        # We run the self-play rollout loop but without returning gradients.
        # collect_batched_self_play_trajectories will play out full games.
        # We need to compute win rate from the info dicts of the final trajectories.
        trajectories = collect_batched_self_play_trajectories(self.num_envs, learning_agent, opponent_agent, self.storage, self.llm_env)
        
        # Analyze trajectories to get metrics
        player_0_wins = 0
        player_1_wins = 0
        draws = 0
        invalid_actions = 0
        total_games = len(trajectories)
        
        if total_games == 0:
            return {"p0_win_rate": 0.0, "p1_win_rate": 0.0, "draw_rate": 0.0, "invalid_action_rate": 0.0}
            
        for traj in trajectories:
            reward = traj.get('reward', 0.0)
            info = traj.get('info', {})
            
            # Simple heuristic based on reward sign (assuming zero-sum)
            if reward > 0:
                player_0_wins += 1
            elif reward < 0:
                if info.get('msg', '').startswith('Illegal') or info.get('msg', '').startswith('Invalid'):
                    invalid_actions += 1
                player_1_wins += 1
            else:
                draws += 1
                
        metrics = {
            "p0_win_rate": player_0_wins / total_games,
            "p1_win_rate": player_1_wins / total_games,
            "draw_rate": draws / total_games,
            "invalid_action_rate": invalid_actions / total_games,
        }
        return metrics

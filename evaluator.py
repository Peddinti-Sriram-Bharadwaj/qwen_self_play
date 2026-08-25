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
        trajectories, game_metrics = collect_batched_self_play_trajectories(self.num_envs, learning_agent, opponent_agent, self.storage, self.llm_env)
        
        return game_metrics

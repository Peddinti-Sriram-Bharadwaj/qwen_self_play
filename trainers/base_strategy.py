from abc import ABC, abstractmethod
from storage import TrajectoryStorage

class TrainerStrategy(ABC):
    """
    GoF Strategy Interface for RL Training Algorithms.
    Every algorithm (PPO, GRPO, DPO, KTO) must implement these methods.
    """
    
    @abstractmethod
    def __init__(self, agent):
        pass

    @abstractmethod
    def collect_data(self, num_envs: int, storage: TrajectoryStorage, llm_env, opponent_agent) -> list:
        """
        Runs the batched rollout loop to collect a batch of trajectories.
        Returns the collected batch of data formatted for the specific algorithm.
        """
        pass

    @abstractmethod
    def update(self, batch: list) -> dict:
        """
        Performs the gradient update using the algorithm-specific loss function.
        Returns a dictionary of stats (Loss, Return, Feature Variance, etc).
        """
        pass

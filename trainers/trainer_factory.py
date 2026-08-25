from trainers.base_strategy import TrainerStrategy

class TrainerFactory:
    """
    GoF Factory Pattern to dynamically instantiate the correct RL training strategy.
    """
    
    @staticmethod
    def get_trainer(algo_name: str, agent) -> TrainerStrategy:
        algo_name = algo_name.upper()
        
        if algo_name == "PPO":
            from trainers.ppo_strategy import PPOStrategy
            return PPOStrategy(agent)
        elif algo_name == "GRPO":
            from trainers.grpo_strategy import GRPOStrategy
            return GRPOStrategy(agent)
        elif algo_name == "REINFORCE":
            from trainers.reinforce_strategy import ReinforceStrategy
            return ReinforceStrategy(agent)
        elif algo_name == "REINFORCE_PLUS":
            from trainers.reinforce_plus_strategy import ReinforcePlusStrategy
            return ReinforcePlusStrategy(agent)
        elif algo_name == "DAPO":
            from trainers.dapo_strategy import DAPOStrategy
            return DAPOStrategy(agent)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}. Supported: PPO, GRPO, REINFORCE, REINFORCE_PLUS, DAPO")

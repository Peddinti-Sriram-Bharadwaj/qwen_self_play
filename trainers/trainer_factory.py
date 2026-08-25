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
        elif algo_name == "DPO":
            from trainers.dpo_strategy import DPOStrategy
            return DPOStrategy(agent)
        elif algo_name == "KTO":
            from trainers.kto_strategy import KTOStrategy
            return KTOStrategy(agent)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}. Supported: PPO, GRPO, DPO, KTO")

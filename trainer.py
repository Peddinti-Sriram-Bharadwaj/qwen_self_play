from trl import PPOTrainer, PPOConfig
import torch

class TRLTrainer:
    def __init__(self, agent):
        self.agent = agent
        
        # PPO Configuration
        # We use a very small batch size for the walking skeleton to run efficiently on Mac
        config = PPOConfig(
            batch_size=2,
            mini_batch_size=2,
            learning_rate=1e-5,
            ppo_epochs=1,
            gradient_accumulation_steps=1,
            # We don't want to optimize the whole model for a walking skeleton to save memory, 
            # but since it's a tiny model we can just let it run.
        )
        
        # PPOTrainer will automatically create a frozen ref_model from model if ref_model=None
        self.ppo_trainer = PPOTrainer(
            config=config,
            model=self.agent.model,
            ref_model=None,
            tokenizer=self.agent.tokenizer
        )

    def update(self, trajectories):
        """
        Extracts queries, responses, and rewards from the trajectories and performs a PPO step.
        """
        queries = []
        responses = []
        rewards = []
        
        for step in trajectories:
            queries.append(step['query'].to(self.agent.device))
            responses.append(step['response'].to(self.agent.device))
            rewards.append(step['reward'].to(self.agent.device))
            
        if len(queries) == 0:
            return {}
            
        # PPO step
        stats = self.ppo_trainer.step(queries, responses, rewards)
        
        print(f"Performed TRL PPO update on {len(queries)} steps.")
        return stats

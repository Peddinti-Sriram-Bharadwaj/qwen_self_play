from trainers.base_strategy import TrainerStrategy
from self_play import collect_batched_self_play_trajectories
from trl import PPOTrainer, PPOConfig
import torch

class PPOStrategy(TrainerStrategy):
    def __init__(self, agent):
        self.agent = agent
        
        # PPO Configuration
        config = PPOConfig(
            learning_rate=1e-5,          # Low LR to prevent degrading the pretrained weights
            batch_size=256,              # Larger batch size for stable gradients (Matches main.py default)
            mini_batch_size=32,          # Must divide batch_size
            gradient_accumulation_steps=1,
            ppo_epochs=4,                # Number of optimisation epochs per batch
            init_kl_coef=0.1,            # KL penalty to keep the model producing valid text
            target_kl=0.1,               # Stop updating if KL diverges too much
            gamma=0.99,                  # Standard discount factor
            lam=0.95,                    # GAE lambda
            cliprange=0.2,               # Standard PPO clipping
            cliprange_value=0.2,         # Value clipping
        )
        
        # PPOTrainer will automatically create a frozen ref_model from model if ref_model=None
        self.ppo_trainer = PPOTrainer(
            config=config,
            model=self.agent.model,
            ref_model=None,
            tokenizer=self.agent.tokenizer
        )

    def collect_data(self, num_envs: int, storage, llm_env) -> list:
        # PPO requires standard batched trajectories
        trajectories = collect_batched_self_play_trajectories(num_envs, self.agent, storage, llm_env)
        
        # Flatten trajectories into a list of steps
        flattened_steps = []
        for traj in trajectories:
            flattened_steps.extend(traj)
            
        return flattened_steps

    def measure_plasticity(self, queries):
        """
        Measures the plasticity of the model based on the current batch of queries.
        """
        self.agent.model.eval()
        with torch.no_grad():
            device = self.agent.model.pretrained_model.device
            input_ids = torch.nn.utils.rnn.pad_sequence(
                queries, batch_first=True, padding_value=self.agent.tokenizer.pad_token_id
            ).to(device)
            
            # Use the underlying HuggingFace model to get hidden states
            outputs = self.agent.model.pretrained_model(input_ids, output_hidden_states=True)
            # Extract last layer hidden states for the last token
            last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32) # (batch, hidden)
            
            # Variance across the batch for each neuron
            neuron_variance = torch.var(last_hidden, dim=0)
            
            # Feature Variance (proxy for rank collapse)
            feature_variance = neuron_variance.mean().item()
            
            # Dormant Neurons
            tolerance = 1e-5
            dormant_neurons = (neuron_variance < tolerance).sum().item()
            total_neurons = last_hidden.shape[-1]
            dormant_percentage = (dormant_neurons / total_neurons) * 100
            
        return feature_variance, dormant_percentage

    def update(self, batch: list) -> dict:
        """
        Extracts queries, responses, and rewards from the batch and performs a PPO step.
        """
        queries = []
        responses = []
        rewards = []
        
        for step in batch:
            queries.append(step['query'].to(self.agent.device))
            responses.append(step['response'].to(self.agent.device))
            rewards.append(step['reward'].to(self.agent.device))
            
        if len(queries) == 0:
            return {}
            
        # Measure plasticity before update
        feat_var, dormant_pct = self.measure_plasticity(queries)
            
        # PPO step
        stats = self.ppo_trainer.step(queries, responses, rewards)
        
        # Inject our plasticity metrics into the stats
        stats['plasticity/feature_variance'] = feat_var
        stats['plasticity/dormant_neurons_pct'] = dormant_pct
        
        return stats

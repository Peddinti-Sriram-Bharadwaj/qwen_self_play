from trainers.base_strategy import TrainerStrategy
from self_play import collect_batched_self_play_trajectories
from trainers.utils import compute_sequence_logprobs
import torch

class ReinforceStrategy(TrainerStrategy):
    """
    REINFORCE with carefully designed running-mean baselines for zero-sum games.
    """
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=1e-5)
        
        # Baseline moving average tracking
        self.baseline = 0.0
        self.baseline_alpha = 0.99

    def collect_data(self, num_envs: int, storage, llm_env) -> list:
        # Standard batched self-play
        trajectories = collect_batched_self_play_trajectories(num_envs, self.agent, storage, llm_env)
        
        flattened_steps = []
        for traj in trajectories:
            flattened_steps.extend(traj)
            
        return flattened_steps

    def update(self, batch: list) -> dict:
        """
        Calculates REINFORCE loss with a baseline.
        """
        self.agent.model.train()
        self.optimizer.zero_grad()
        
        queries = [step['query'].to(self.agent.device) for step in batch]
        responses = [step['response'].to(self.agent.device) for step in batch]
        rewards = torch.stack([step['reward'].to(self.agent.device) for step in batch])
        
        if len(queries) == 0:
            return {}
            
        # Update running baseline
        batch_mean = rewards.mean().item()
        self.baseline = (self.baseline_alpha * self.baseline) + ((1 - self.baseline_alpha) * batch_mean)
        
        # Calculate advantages (Reward - Baseline)
        advantages = rewards - self.baseline
        
        # We need the pad_token_id for masking
        pad_token_id = self.agent.tokenizer.pad_token_id
        
        # Calculate log probs and loss for each step
        losses = []
        for i in range(len(batch)):
            log_prob = compute_sequence_logprobs(self.agent.model, queries[i], responses[i], pad_token_id)
            # REINFORCE objective: maximize expected return -> minimize -log_prob * advantage
            step_loss = -log_prob * advantages[i]
            losses.append(step_loss)
            
        total_loss = torch.stack(losses).mean()
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1.0)
        self.optimizer.step()
        
        print(f"Performed REINFORCE update on {len(batch)} steps. Loss: {total_loss.item():.4f}, Baseline: {self.baseline:.4f}")
        
        return {
            "reinforce/loss": total_loss.item(),
            "reinforce/baseline": self.baseline,
            "plasticity/feature_variance": 0.0, 
            "plasticity/dormant_neurons_pct": 0.0
        }

from trainers.base_strategy import TrainerStrategy
from self_play import collect_batched_self_play_trajectories
from trainers.utils import compute_sequence_logprobs, calculate_plasticity_metrics, compute_policy_entropy
import torch

class DAPOStrategy(TrainerStrategy):
    """
    DAPO-like Strategy.
    Useful for sampling, clipping, and entropy-preservation improvements.
    """
    def __init__(self, agent, lr: float = 1e-5, beta: float = 0.1, **kwargs):
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=lr)
        self.beta = beta

    def collect_data(self, num_envs: int, storage, llm_env, opponent_agent) -> list:
        trajectories, game_metrics = collect_batched_self_play_trajectories(num_envs, self.agent, opponent_agent, storage, llm_env)
        
        flattened_steps = []
        for traj in trajectories:
            flattened_steps.extend(traj)
            
        return flattened_steps, game_metrics

    def update(self, batch: list) -> dict:
        """
        Calculates DAPO loss with entropy preservation.
        """
        self.agent.model.train()
        self.optimizer.zero_grad()
        
        queries = [step['query'].to(self.agent.device) for step in batch]
        responses = [step['response'].to(self.agent.device) for step in batch]
        rewards = torch.stack([step['reward'].to(self.agent.device) for step in batch])
        
        if len(queries) == 0:
            return {}
            
        # Standardize rewards (Advantages)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        pad_token_id = self.agent.tokenizer.pad_token_id
        
        total_loss_val = 0.0
        total_log_prob = 0.0
        total_entropy = 0.0
        clip_ratio = 0.2
        beta = self.beta # Configurable entropy penalty coefficient
        
        for i in range(len(batch)):
            log_prob = compute_sequence_logprobs(self.agent.model, queries[i], responses[i], pad_token_id)
            
            # Compute policy entropy for monitoring exploration hacking
            step_entropy = compute_policy_entropy(self.agent.model, queries[i], pad_token_id)
            total_entropy += step_entropy
            
            # For a full DAPO, we would maintain a reference model to get ref_log_probs
            # and calculate the probability ratio. Here we use a simplified version:
            # We assume old_log_prob is just detached current log_prob for the first epoch
            old_log_prob = log_prob.detach()
            
            ratio = torch.exp(log_prob - old_log_prob)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[i]
            
            # Entropy preservation (simplified by penalizing overly confident log_probs)
            entropy_penalty = beta * log_prob
            
            step_loss = -torch.min(surr1, surr2) - entropy_penalty
            
            scaled_loss = step_loss / len(batch)
            scaled_loss.backward()
            
            total_loss_val += step_loss.item()
            total_log_prob += log_prob.item()
            
        avg_loss = total_loss_val / len(batch)
        
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1.0)
        self.optimizer.step()
        
        
        avg_reward = rewards.mean().item()
        win_rate = (rewards > 0).float().mean().item()
        avg_seq_len = sum(len(r) for r in responses) / len(responses)
        avg_log_prob = total_log_prob / len(batch)
        avg_entropy = total_entropy / len(batch)
        
        # Calculate Plasticity Metrics
        latents_batch = [step['latents'].to(self.agent.device) for step in batch if 'latents' in step]
        plasticity_metrics = calculate_plasticity_metrics(latents_batch)
        
        print(f"Performed DAPO update on {len(batch)} steps. Loss: {avg_loss:.4f} | Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f} | Entropy: {avg_entropy:.2f}")
        
        stats = {
            "dapo/loss": avg_loss,
            "dapo/mean_reward": avg_reward,
            "dapo/win_rate": win_rate,
            "dapo/avg_seq_len": avg_seq_len,
            "dapo/mean_log_prob": avg_log_prob,
            "dapo/policy_entropy": avg_entropy,
        }
        stats.update(plasticity_metrics)
        return stats

from trainers.base_strategy import TrainerStrategy
from self_play import collect_batched_self_play_trajectories
from trainers.utils import compute_sequence_logprobs
import torch

class ReinforcePlusStrategy(TrainerStrategy):
    """
    REINFORCE++ / TRR++ Strategy.
    Useful for task- and role-dependent proposer-solver setups.
    """
    def __init__(self, agent):
        self.agent = agent
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=1e-5)

    def collect_data(self, num_envs: int, storage, llm_env, opponent_agent) -> list:
        # In a Proposer-Solver setup, we might split the roles across the batch.
        trajectories = collect_batched_self_play_trajectories(num_envs, self.agent, opponent_agent, storage, llm_env)
        
        flattened_steps = []
        for traj in trajectories:
            flattened_steps.extend(traj)
            
        return flattened_steps

    def update(self, batch: list) -> dict:
        """
        Calculates REINFORCE++ loss.
        """
        self.agent.model.train()
        self.optimizer.zero_grad()
        
        queries = [step['query'].to(self.agent.device) for step in batch]
        responses = [step['response'].to(self.agent.device) for step in batch]
        rewards = torch.stack([step['reward'].to(self.agent.device) for step in batch])
        
        if len(queries) == 0:
            return {}
            
        # REINFORCE++ often centers rewards per batch or uses a task-specific advantage
        baseline = rewards.mean()
        advantages = rewards - baseline
        
        pad_token_id = self.agent.tokenizer.pad_token_id
        
        total_loss_val = 0.0
        total_log_prob = 0.0
        for i in range(len(batch)):
            log_prob = compute_sequence_logprobs(self.agent.model, queries[i], responses[i], pad_token_id)
            
            # Additional TRR++ logic could weight positive advantages more heavily
            # or apply specific role-based scaling. Here we apply a simple positive-advantage boost
            adv = advantages[i]
            if adv > 0:
                adv = adv * 1.5 # Boost successful solver/proposer paths
                
            step_loss = -log_prob * adv
            
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
        
        print(f"Performed REINFORCE++ update on {len(batch)} steps. Loss: {avg_loss:.4f} | Reward: {avg_reward:.2f} | Win Rate: {win_rate:.2f}")
        
        return {
            "reinforce_plus/loss": avg_loss,
            "reinforce_plus/mean_reward": avg_reward,
            "reinforce_plus/win_rate": win_rate,
            "reinforce_plus/avg_seq_len": avg_seq_len,
            "reinforce_plus/mean_log_prob": avg_log_prob,
            "plasticity/feature_variance": 0.0, 
            "plasticity/dormant_neurons_pct": 0.0
        }

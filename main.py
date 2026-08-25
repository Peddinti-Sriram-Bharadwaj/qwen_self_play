import torch
from environment import TicTacToeEnv
from agent import LocalLLMAgent
from self_play import collect_self_play_trajectory
from storage import TrajectoryStorage
from trainer import TRLTrainer

def main():
    print("=== Walking Skeleton: Continual RL with Self-Play (TRL Integration) ===")
    print("Initializing environment (Tic-Tac-Toe)...")
    env = TicTacToeEnv()
    
    # We use a small local model for the walking skeleton. 
    # Qwen 3.6 can be dropped-in here later.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Initializing agent with {model_name}...")
    # For self-play we can use the same agent playing against itself (shared weights)
    agent = LocalLLMAgent(model_name=model_name)
    
    print("Initializing TRL PPOTrainer...")
    trainer = TRLTrainer(agent)
    
    print("Initializing Trajectory Storage...")
    storage = TrajectoryStorage(base_dir="latents")
    
    num_iterations = 100 # Overall training loops
    episodes_per_iteration = 10 # Collect 10 games before trying to train
    
    all_trajectories = []
    
    for iteration in range(num_iterations):
        print(f"\n========== Training Iteration {iteration+1}/{num_iterations} ==========")
        
        # 1. Collect phase
        for episode in range(episodes_per_iteration):
            trajectory = collect_self_play_trajectory(env, agent, agent, storage, render=False)
            all_trajectories.extend(trajectory)
            
        print(f"Collected {len(all_trajectories)} steps from {episodes_per_iteration} episodes.")
        
        # 2. Train phase
        batch_size = trainer.ppo_trainer.config.batch_size
        
        if len(all_trajectories) >= batch_size:
            # We train on as many full batches as we collected
            num_batches = len(all_trajectories) // batch_size
            for b in range(num_batches):
                start_idx = b * batch_size
                batch = all_trajectories[start_idx : start_idx + batch_size]
                stats = trainer.update(batch)
                
            print(f"Trainer Stats (Last Batch):")
            print(f"  - Policy Loss: {stats.get('ppo/loss/policy', 'N/A')}")
            print(f"  - Value Loss: {stats.get('ppo/loss/value', 'N/A')}")
            print(f"  - KL Divergence: {stats.get('objective/kl', 'N/A')}")
            print(f"  - Return: {stats.get('ppo/returns/mean', 'N/A')}")
            print(f"Plasticity Metrics:")
            print(f"  - Feature Variance: {stats.get('plasticity/feature_variance', 'N/A')}")
            print(f"  - Dormant Neurons (%): {stats.get('plasticity/dormant_neurons_pct', 'N/A')}")
            
            # Keep leftovers for the next iteration
            remainder = len(all_trajectories) % batch_size
            if remainder > 0:
                all_trajectories = all_trajectories[-remainder:]
            else:
                all_trajectories = []
        else:
            print(f"Not enough trajectories ({len(all_trajectories)}) for a batch ({batch_size}). Accumulating...")
            
        # 3. Checkpointing (for Safety Frontier Evaluation)
        if (iteration + 1) % 10 == 0:
            import os
            checkpoint_dir = f"checkpoints/iter_{iteration+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Saving model checkpoint to {checkpoint_dir}...")
            # We save the active policy model
            agent.model.pretrained_model.save_pretrained(checkpoint_dir)
            agent.tokenizer.save_pretrained(checkpoint_dir)

if __name__ == "__main__":
    main()

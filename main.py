import torch
from environment import TicTacToeEnv
from agent import LocalLLMAgent
from self_play import collect_self_play_trajectory
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
    
    num_episodes = 2
    all_trajectories = []
    
    # 1. Collect phase
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        trajectory = collect_self_play_trajectory(env, agent, agent, render=True)
        all_trajectories.extend(trajectory)
        print(f"Collected trajectory of length {len(trajectory)}.")
        
    # 2. Train phase
    # TRL PPO performs best with batched updates, so we pool all steps from the episodes
    print(f"\n--- Training Phase ---")
    print(f"Total steps accumulated: {len(all_trajectories)}")
    
    # We must ensure we feed batch_size elements to the trainer, or it will throw an error.
    # The config expects batch_size=2. Let's slice it just to be safe.
    batch_size = trainer.ppo_trainer.config.batch_size
    
    if len(all_trajectories) >= batch_size:
        # For the skeleton, just grab a batch and train
        batch = all_trajectories[:batch_size]
        stats = trainer.update(batch)
        
        print(f"Trainer Stats:")
        print(f"  - Policy Loss: {stats.get('ppo/loss/policy', 'N/A')}")
        print(f"  - Value Loss: {stats.get('ppo/loss/value', 'N/A')}")
        print(f"  - KL Divergence: {stats.get('objective/kl', 'N/A')}")
        print(f"  - Return: {stats.get('ppo/returns/mean', 'N/A')}")
    else:
        print("Not enough trajectories collected for a batch update.")

if __name__ == "__main__":
    main()

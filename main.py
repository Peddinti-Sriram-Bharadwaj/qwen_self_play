import torch
from environment import TicTacToeEnv
from agent import LocalLLMAgent
from self_play import collect_self_play_trajectory
from trainer import PPOTrainer

def main():
    print("=== Walking Skeleton: Continual RL with Self-Play ===")
    print("Initializing environment (Tic-Tac-Toe)...")
    env = TicTacToeEnv()
    
    # We use a small local model for the walking skeleton. 
    # Qwen 3.6 can be dropped-in here later.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Initializing agent with {model_name}...")
    # For self-play we can use the same agent playing against itself (shared weights)
    agent = LocalLLMAgent(model_name=model_name)
    
    print("Initializing Trainer...")
    trainer = PPOTrainer(agent)
    
    num_episodes = 2
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        trajectory = collect_self_play_trajectory(env, agent, agent, render=True)
        
        print(f"Collected trajectory of length {len(trajectory)}.")
        
        loss = trainer.update(trajectory)
        print(f"Trainer average loss for episode: {loss:.4f}")

if __name__ == "__main__":
    main()

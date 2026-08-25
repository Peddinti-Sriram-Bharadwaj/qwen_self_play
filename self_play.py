from environment import TicTacToeEnv
from agent import Agent
from storage import TrajectoryStorage
import torch

def collect_self_play_trajectory(env: TicTacToeEnv, agent1: Agent, agent2: Agent, storage: TrajectoryStorage, render: bool = False):
    obs = env.reset()
    done = False
    
    # We will store: (query_tensor, response_tensor, reward)
    trajectory = []
    
    agents = {1: agent1, -1: agent2}
    
    storage.start_new_episode()
    
    if render:
        print("Starting new self-play episode...")
    
    while not done:
        current_player = env.current_player
        agent = agents[current_player]
        
        if render:
            print(f"Player {'X' if current_player == 1 else 'O'}'s turn.")
            
        # 1. Agent acts
        action, query_tensor, response_tensor, cot_latents = agent.act(obs)
        if render:
            print(f"Chosen action: {action}")
            
        # 1.5 Flush latents to disk
        storage.save_step_latents(current_player, cot_latents)
            
        # 2. Step environment
        next_obs, reward, done, info = env.step(action)
        
        if render:
            print(f"Reward: {reward}")
            print("-" * 20)
            
        trajectory.append({
            'player': current_player,
            'query': query_tensor,
            'response': response_tensor,
            'reward': torch.tensor(reward, dtype=torch.float32)
        })
        
        obs = next_obs
        
    if render:
        print(f"Episode finished. Info: {info}")
        
    # --- Episodic Credit Assignment ---
    # TRL's PPO step treats each query-response pair independently. 
    # To learn strategy, all moves in a winning game must be rewarded, and all moves in a losing game penalized.
    
    # If the game ended due to a valid terminal state (win/draw)
    if "winner" in info:
        winner = info["winner"]
        for step in trajectory:
            if winner is None:
                step['reward'] = torch.tensor(0.0, dtype=torch.float32) # Draw
            elif step['player'] == winner:
                step['reward'] = torch.tensor(1.0, dtype=torch.float32) # Win
            else:
                step['reward'] = torch.tensor(-1.0, dtype=torch.float32) # Loss
                
    # If the game ended due to an invalid move, the last step already has -10.0.
    # We optionally penalize all previous moves by that player too, but usually just penalizing the mistake is enough.
        
    return trajectory

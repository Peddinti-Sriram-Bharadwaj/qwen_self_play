from environment import TicTacToeEnv
from agent import Agent

def collect_self_play_trajectory(env: TicTacToeEnv, agent1: Agent, agent2: Agent, render: bool = False):
    obs = env.reset()
    done = False
    
    # We will store: (observation, action, reward, latents)
    trajectory = []
    
    agents = {1: agent1, -1: agent2}
    
    if render:
        print("Starting new self-play episode...")
    
    while not done:
        current_player = env.current_player
        agent = agents[current_player]
        
        if render:
            print(f"Player {'X' if current_player == 1 else 'O'}'s turn.")
            
        # 1. Agent acts
        action = agent.act(obs)
        if render:
            print(f"Chosen action: {action}")
            
        # 2. Extract latents for analysis or PPO
        latents = agent.extract_latents(obs)
        
        # 3. Step environment
        next_obs, reward, done, info = env.step(action)
        
        if render:
            print(f"Reward: {reward}")
            print("-" * 20)
            
        trajectory.append({
            'player': current_player,
            'observation': obs,
            'action': action,
            'reward': reward,
            'latents': latents,
            'next_observation': next_obs,
            'done': done,
            'info': info
        })
        
        obs = next_obs
        
    if render:
        print(f"Episode finished. Info: {info}")
        
    return trajectory

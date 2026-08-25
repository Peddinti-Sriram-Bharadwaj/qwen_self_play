from agent import Agent
from storage import TrajectoryStorage
from envs.env_factory import EnvFactory
import torch
import copy

def collect_batched_self_play_trajectories(num_envs: int, agent: Agent, storage: TrajectoryStorage, llm_env):
    envs = [copy.deepcopy(llm_env) for _ in range(num_envs)]
    obs_list = [env.reset(player_id=1) for env in envs]
    
    # Track state for each active environment
    active_trajectories = [[] for _ in range(num_envs)]
    episode_ids = [storage.start_new_episode() for _ in range(num_envs)]
    step_counters = [0 for _ in range(num_envs)]
    
    # For invalid action resampling
    invalid_action_retries = [0 for _ in range(num_envs)]
    MAX_RETRIES = 2
    
    completed_trajectories = []
    
    # We run until all initial N environments finish at least one game.
    # To keep the batch size exactly N, if an env finishes early, we reset it and keep collecting.
    # We will stop when we have collected exactly `num_envs` completed trajectories.
    
    while len(completed_trajectories) < num_envs:
        # 1. Agent acts on all observations simultaneously
        actions, query_tensors, response_tensors, cot_latents_batch = agent.batched_act(obs_list)
        
        # 2. Step all environments
        for i in range(num_envs):
            env = envs[i]
            # Assuming player 1 for turn tracking if the env doesn't expose it natively yet
            current_player = 1 
            text_action = actions[i]
            
            # Save latents
            step_counters[i] += 1
            storage.save_step_latents(episode_ids[i], step_counters[i], current_player, cot_latents_batch[i])
            
            # Step env (this will handle parsing internally)
            next_obs, reward, done, info = env.step(text_action)
            
            # Invalid Action Resampling Logic
            if next_obs == "INVALID":
                invalid_action_retries[i] += 1
                if invalid_action_retries[i] <= MAX_RETRIES:
                    # Resample: Append a warning to the observation to prompt the model again
                    obs_list[i] = obs_list[i] + "\nWARNING: Your previous action was invalid. Please try again. End with ACTION: <move>"
                    # Do not append to active_trajectories or step forward
                    continue
                else:
                    # Give up, penalize heavily, and end the game
                    reward = -10.0
                    done = True
                    info["winner"] = -1 # Opponent wins by default
                    invalid_action_retries[i] = 0
            else:
                # Valid move, reset retries
                invalid_action_retries[i] = 0
            
            active_trajectories[i].append({
                'player': current_player,
                'query': query_tensors[i],
                'response': response_tensors[i],
                'reward': torch.tensor(reward, dtype=torch.float32)
            })
            
            if done:
                # --- Episodic Credit Assignment ---
                trajectory = active_trajectories[i]
                if "winner" in info:
                    winner = info["winner"]
                    for step in trajectory:
                        if winner is None:
                            step['reward'] = torch.tensor(0.0, dtype=torch.float32) # Draw
                        elif step['player'] == winner:
                            step['reward'] = torch.tensor(1.0, dtype=torch.float32) # Win
                        else:
                            step['reward'] = torch.tensor(-1.0, dtype=torch.float32) # Loss
                            
                completed_trajectories.append(trajectory)
                
                # Reset environment for the next game to keep batch full
                obs_list[i] = env.reset(player_id=1)
                active_trajectories[i] = []
                episode_ids[i] = storage.start_new_episode()
                step_counters[i] = 0
                invalid_action_retries[i] = 0
            else:
                obs_list[i] = next_obs
                
    # We might have collected more than num_envs if some finished super fast, 
    # but returning all completed is fine for PPO.
    return completed_trajectories

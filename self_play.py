from agent import Agent
from storage import TrajectoryStorage
from envs.env_factory import EnvFactory
import torch
import copy

def collect_batched_self_play_trajectories(num_envs: int, agent: Agent, opponent_agent: Agent, storage: TrajectoryStorage, llm_env):
    envs = [copy.deepcopy(llm_env) for _ in range(num_envs)]
    
    obs_list = []
    current_players = []
    for env in envs:
        o, p = env.reset()
        obs_list.append(o)
        current_players.append(p)
        
    # Track state for each active environment
    active_trajectories = [[] for _ in range(num_envs)]
    episode_ids = [storage.start_new_episode() for _ in range(num_envs)]
    step_counters = [0 for _ in range(num_envs)]
    
    # For invalid action resampling
    invalid_action_retries = [0 for _ in range(num_envs)]
    MAX_RETRIES = 2
    
    completed_trajectories = []
    
    while len(completed_trajectories) < num_envs:
        learner_indices = [i for i, p in enumerate(current_players) if p == 0]
        opponent_indices = [i for i, p in enumerate(current_players) if p == 1]
        
        actions = [None] * num_envs
        query_tensors = [None] * num_envs
        response_tensors = [None] * num_envs
        cot_latents_batch = [None] * num_envs
        
        # 1a. Learner acts (Player 0)
        if learner_indices:
            l_obs = [obs_list[i] for i in learner_indices]
            l_actions, l_queries, l_responses, l_cot = agent.batched_act(l_obs)
            for j, i in enumerate(learner_indices):
                actions[i] = l_actions[j]
                query_tensors[i] = l_queries[j]
                response_tensors[i] = l_responses[j]
                cot_latents_batch[i] = l_cot[j]
                
        # 1b. Opponent acts (Player 1)
        if opponent_indices:
            o_obs = [obs_list[i] for i in opponent_indices]
            o_actions, o_queries, o_responses, o_cot = opponent_agent.batched_act(o_obs)
            for j, i in enumerate(opponent_indices):
                actions[i] = o_actions[j]
                query_tensors[i] = o_queries[j]
                response_tensors[i] = o_responses[j]
                cot_latents_batch[i] = o_cot[j]
                
        # 2. Step all environments
        for i in range(num_envs):
            env = envs[i]
            current_player = current_players[i]
            text_action = actions[i]
            
            # Save latents only for the learning agent (Player 0)
            if current_player == 0:
                step_counters[i] += 1
                storage.save_step_latents(episode_ids[i], step_counters[i], current_player, cot_latents_batch[i])
            
            # Step env
            next_obs, reward, done, info, next_player = env.step(text_action)
            
            # Invalid Action Resampling Logic
            if next_obs == "INVALID":
                invalid_action_retries[i] += 1
                if invalid_action_retries[i] <= MAX_RETRIES:
                    # Resample: Append a warning
                    obs_list[i] = obs_list[i] + "\nWARNING: Your previous action was invalid. Please try again. End with ACTION: <move>"
                    continue
                else:
                    # Give up, penalize heavily, and end the game
                    reward = -10.0
                    done = True
                    info["returns"] = [-10.0, 10.0] if current_player == 0 else [10.0, -10.0]
                    invalid_action_retries[i] = 0
            else:
                invalid_action_retries[i] = 0
            
            # Only append to active_trajectories if it was the Learner's action
            # We want to train the Learner's policy!
            if current_player == 0:
                active_trajectories[i].append({
                    'player': current_player,
                    'query': query_tensors[i],
                    'response': response_tensors[i],
                    'reward': torch.tensor(reward, dtype=torch.float32)
                })
            
            if done:
                # --- Episodic Credit Assignment ---
                trajectory = active_trajectories[i]
                
                # Retrieve returns for the Learner (Player 0)
                # OpenSpiel/PettingZoo returns a list or dict of rewards.
                if "returns" in info:
                    if isinstance(info["returns"], (list, tuple)):
                        final_reward = info["returns"][0]
                    elif isinstance(info["returns"], dict):
                        # Attempt to get player 0's reward
                        p0_key = next((k for k in info["returns"].keys() if "0" in str(k)), 0)
                        final_reward = info["returns"].get(p0_key, reward)
                    else:
                        final_reward = reward
                        
                    for step in trajectory:
                        step['reward'] = torch.tensor(final_reward, dtype=torch.float32)
                            
                completed_trajectories.append(trajectory)
                
                # Reset environment
                o, p = env.reset()
                obs_list[i] = o
                current_players[i] = p
                active_trajectories[i] = []
                episode_ids[i] = storage.start_new_episode()
                step_counters[i] = 0
                invalid_action_retries[i] = 0
            else:
                obs_list[i] = next_obs
                current_players[i] = next_player
                
    return completed_trajectories

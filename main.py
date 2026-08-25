import os
import argparse

# Parse args before importing torch/jax to configure OS-level GPU isolation
parser = argparse.ArgumentParser(description="Generalized LLM MARL Training")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "GRPO", "REINFORCE", "REINFORCE_PLUS", "DAPO"], help="RL Algorithm to run")
parser.add_argument("--env", type=str, default="TicTacToe-v0", help="The name of the environment")
parser.add_argument("--backend", type=str, default="textarena", choices=["textarena", "openspiel", "pettingzoo", "jaxmarl"], help="The environment suite backend")
parser.add_argument("--llm-gpu", type=int, default=0, help="GPU ID for the LLM")
parser.add_argument("--env-gpu", type=int, default=1, help="GPU ID for JaxMARL")
parser.add_argument("--batch-size", type=int, default=256, help="Global batch size for RL updates")
parser.add_argument("--num-envs", type=int, default=32, help="Number of parallel environments to run during collection")
parser.add_argument("--regime", type=str, default="B", choices=["A", "B", "C", "D"], help="Self-Play Regime (A: Fixed, B: Evolving, C: High Replay, D: League)")
args = parser.parse_args()

# Disable JAX preallocation immediately so it doesn't crash PyTorch LLMs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import torch
from environment import TicTacToeEnv
from agent import LocalLLMAgent
from self_play import collect_batched_self_play_trajectories
from storage import TrajectoryStorage
from trainers.trainer_factory import TrainerFactory
from envs.env_factory import EnvFactory
from opponents import OpponentManager
import argparse

def main():
    print(f"=== Generalized LLM MARL: {args.algo} on {args.backend.upper()} ({args.env}) ===")
    
    # env initialization is now handled dynamically in self_play.py for vectorized rollouts
    
    # We use a small local model for the walking skeleton. 
    # Qwen 3.6 can be dropped-in here later.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Hardware Partitioning for JaxMARL
    if args.backend == "jaxmarl":
        print(f"Hardware Partitioning: Pinning Qwen to GPU {args.llm_gpu}, JaxMARL to GPU {args.env_gpu}...")
        try:
            import jax
            if len(jax.devices()) > args.env_gpu:
                jax.config.update('jax_default_device', jax.devices()[args.env_gpu])
        except Exception:
            pass
            
    # Assign the LLM explicitly to the correct GPU
    if torch.cuda.is_available():
        device = f"cuda:{args.llm_gpu}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Initializing agent with {model_name} on {device}...")
    agent = LocalLLMAgent(model_name=model_name, device=device)
    
    print(f"Initializing {args.backend.upper()} Environment wrapper...")
    # Base prototype of EnvFactory usage
    llm_env = EnvFactory.get_env(backend=args.backend, env_name=args.env)
    
    print(f"Initializing {args.algo} Trainer via Factory...")
    trainer = TrainerFactory.get_trainer(args.algo, agent)
    
    print("Initializing Trajectory Storage...")
    storage = TrajectoryStorage(base_dir=f"latents_{args.algo.lower()}_{args.env.replace('-', '_')}")
    
    print(f"Initializing Opponent Manager for Regime {args.regime}...")
    opponent_manager = OpponentManager(regime=args.regime, learning_agent=agent, checkpoint_dir=f"checkpoints_pool_{args.algo.lower()}")
    
    num_iterations = 1000 # Overall training loops (Raised to force plasticity loss)
    # 32 concurrent games running in parallel (saturating the GPU during rollout)
    num_envs = args.num_envs
    batch_size = args.batch_size # Larger batch size for stable gradients
    
    replay_buffer = []
    
    for iteration in range(num_iterations):
        print(f"\n========== Training Iteration {iteration+1}/{num_iterations} ==========")
        # Prepare the opponent for this batch (Loads weights if FSP)
        opponent_manager.prepare_batch_opponent()
        
        # 1. Collect phase (Delegated to Strategy, passing the LLMEnvironment and opponent)
        trajectories = trainer.collect_data(num_envs, storage, llm_env=llm_env, opponent_agent=opponent_manager.get_opponent())
        replay_buffer.extend(trajectories)
            
        print(f"Collected {len(trajectories)} steps. Buffer size: {len(replay_buffer)}")
        
        # 2. Train phase
        
        # High Replay Buffer maintenance for Regime C
        if args.regime == "C" and len(replay_buffer) > 50000:
            replay_buffer = replay_buffer[-50000:]
            
        if len(replay_buffer) >= batch_size:
            # We train on as many full batches as we can
            num_batches = len(replay_buffer) // batch_size
            
            for b in range(num_batches):
                if args.regime == "C":
                    import random
                    # Sample uniformly from the high replay buffer
                    batch = random.sample(replay_buffer, batch_size)
                else:
                    # On-policy: just take the oldest collected batch in the queue
                    start_idx = b * batch_size
                    batch = replay_buffer[start_idx : start_idx + batch_size]
                    
                stats = trainer.update(batch)
                
            print(f"Trainer Stats ({args.algo} - Last Batch):")
            for k, v in stats.items():
                if not k.startswith("plasticity"):
                    print(f"  - {k}: {v}")
            print(f"Plasticity Metrics:")
            print(f"  - Feature Variance: {stats.get('plasticity/feature_variance', 'N/A')}")
            print(f"  - Dormant Neurons (%): {stats.get('plasticity/dormant_neurons_pct', 'N/A')}")
            
            # If not using High Replay, clear the used on-policy data
            if args.regime != "C":
                remainder = len(replay_buffer) % batch_size
                if remainder > 0:
                    replay_buffer = replay_buffer[-remainder:]
                else:
                    replay_buffer = []
        else:
            print(f"Not enough trajectories ({len(replay_buffer)}) for a batch ({batch_size}). Accumulating...")
            
        # 3. Checkpointing (for Safety Frontier Evaluation and League Training)
        if (iteration + 1) % 10 == 0:
            import os
            checkpoint_dir = f"checkpoints_{args.algo.lower()}/iter_{iteration+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Saving {args.algo} model checkpoint to {checkpoint_dir}...")
            # We save the active policy model
            agent.model.pretrained_model.save_pretrained(checkpoint_dir)
            agent.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save into opponent pool for Regime D
            opponent_manager.save_checkpoint(iteration + 1)

if __name__ == "__main__":
    main()

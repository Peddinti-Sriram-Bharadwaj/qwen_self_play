import torch
from environment import TicTacToeEnv
from agent import LocalLLMAgent
from self_play import collect_batched_self_play_trajectories
from storage import TrajectoryStorage
from trainers.trainer_factory import TrainerFactory
import argparse

def main():
    parser = argparse.ArgumentParser(description="Continual RL Multi-Algorithm Training")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "GRPO", "DPO", "KTO"], help="RL Algorithm to run")
    args = parser.parse_args()
    
    print(f"=== Walking Skeleton: Continual RL with Self-Play ({args.algo} Integration) ===")
    
    # env initialization is now handled dynamically in self_play.py for vectorized rollouts
    
    # We use a small local model for the walking skeleton. 
    # Qwen 3.6 can be dropped-in here later.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print(f"Initializing agent with {model_name}...")
    # For self-play we can use the same agent playing against itself (shared weights)
    agent = LocalLLMAgent(model_name=model_name)
    
    print(f"Initializing {args.algo} Trainer via Factory...")
    trainer = TrainerFactory.get_trainer(args.algo, agent)
    
    print("Initializing Trajectory Storage...")
    storage = TrajectoryStorage(base_dir=f"latents_{args.algo.lower()}")
    
    num_iterations = 500 # Overall training loops (Raised to force plasticity loss)
    # 32 concurrent games running in parallel (saturating the GPU)
    num_envs = 32
    batch_size = 32 # Common batch size across algos
    
    all_trajectories = []
    
    for iteration in range(num_iterations):
        print(f"\n========== Training Iteration {iteration+1}/{num_iterations} ==========")
        # 1. Collect phase (Delegated to Strategy)
        trajectories = trainer.collect_data(num_envs, storage)
        all_trajectories.extend(trajectories)
            
        print(f"Collected {len(all_trajectories)} steps from batched rollouts.")
        
        # 2. Train phase
        
        if len(all_trajectories) >= batch_size:
            # We train on as many full batches as we collected
            num_batches = len(all_trajectories) // batch_size
            for b in range(num_batches):
                start_idx = b * batch_size
                batch = all_trajectories[start_idx : start_idx + batch_size]
                stats = trainer.update(batch)
                
            print(f"Trainer Stats ({args.algo} - Last Batch):")
            for k, v in stats.items():
                if not k.startswith("plasticity"):
                    print(f"  - {k}: {v}")
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
            checkpoint_dir = f"checkpoints_{args.algo.lower()}/iter_{iteration+1}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Saving {args.algo} model checkpoint to {checkpoint_dir}...")
            # We save the active policy model
            agent.model.pretrained_model.save_pretrained(checkpoint_dir)
            agent.tokenizer.save_pretrained(checkpoint_dir)

if __name__ == "__main__":
    main()

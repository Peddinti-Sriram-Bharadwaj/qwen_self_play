import os
import argparse

# Parse the GPU argument BEFORE importing PyTorch
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0", help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES)")
_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _args.gpu

import wandb
from code_self_play import GRPOSelfPlayLoop
from code_agent import DualLoraCodeAgent
from evaluate import evaluate_checkpoint

def run_experiment():
    parser = argparse.ArgumentParser(description="Strict Multi-Phase Catastrophic Forgetting Experiment")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--run-name", type=str, default="strict_forgetting_eval")
    parser.add_argument("--phase_steps", type=int, default=200, help="Steps per phase")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--G", type=int, default=4)
    args = parser.parse_args()

    wandb.init(project="anchored_code_self_play", name=args.run_name, config={"beta": args.beta, "K": args.K, "G": args.G})

    print("Initializing Dual LoRA Agent...")
    agent = DualLoraCodeAgent(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    
    print("\n--- PHASE 0: Baseline Evaluation ---")
    base_eval_A = evaluate_checkpoint("base", "data/eval_A.json", agent)
    base_eval_B = evaluate_checkpoint("base", "data/eval_B.json", agent)
    
    # We log these baselines continuously so they form a horizontal line in wandb
    def log_metrics(step, phase_name, train_metrics):
        metrics = {
            f"{phase_name}/fixer_loss": train_metrics["fixer_loss"],
            f"{phase_name}/generator_loss": train_metrics["generator_loss"],
            f"{phase_name}/fixer_success_rate": train_metrics["fixer_success_rate"],
            f"{phase_name}/generator_reward": train_metrics["generator_reward"],
            "baseline/eval_A_pass_rate": base_eval_A,
            "baseline/eval_B_pass_rate": base_eval_B
        }
        wandb.log(metrics, step=step)

    def run_evals(step, checkpoint_path):
        eval_A = evaluate_checkpoint(checkpoint_path, "data/eval_A.json", agent)
        eval_B = evaluate_checkpoint(checkpoint_path, "data/eval_B.json", agent)
        wandb.log({
            "eval/eval_A_pass_rate": eval_A,
            "eval/eval_B_pass_rate": eval_B
        }, step=step)
        
    print(f"\n--- PHASE 1: Training on Distribution A ({args.phase_steps} steps) ---")
    loop_A = GRPOSelfPlayLoop(agent=agent, dataset_path="data/train_A.json", lr=1e-5, beta=args.beta)
    
    for step in range(1, args.phase_steps + 1):
        metrics = loop_A.run_step(G=args.G, K=args.K)
        if metrics:
            log_metrics(step, "Phase_A", metrics)
            
        if step % args.eval_freq == 0:
            ckpt_path = f"checkpoints/phaseA_step_{step}_fixer"
            agent.set_active_role("fixer")
            agent.model.save_pretrained(ckpt_path)
            run_evals(step, ckpt_path)

    print(f"\n--- PHASE 2: Training on Distribution B ({args.phase_steps} steps) ---")
    # Switch dataset to B
    loop_B = GRPOSelfPlayLoop(agent=agent, dataset_path="data/train_B.json", lr=1e-5, beta=args.beta)
    
    for step in range(args.phase_steps + 1, (args.phase_steps * 2) + 1):
        metrics = loop_B.run_step(G=args.G, K=args.K)
        if metrics:
            log_metrics(step, "Phase_B", metrics)
            
        if step % args.eval_freq == 0:
            ckpt_path = f"checkpoints/phaseB_step_{step}_fixer"
            agent.set_active_role("fixer")
            agent.model.save_pretrained(ckpt_path)
            run_evals(step, ckpt_path)
            
    print("\n--- EXPERIMENT COMPLETE ---")

if __name__ == "__main__":
    run_experiment()

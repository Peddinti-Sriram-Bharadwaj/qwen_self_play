import os
import argparse

# Parse the GPU argument BEFORE importing PyTorch to guarantee strict VRAM isolation
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0", help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES)")
_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _args.gpu

import wandb
from code_self_play import CoderTesterSelfPlayLoop, GRPOSelfPlayLoop
from code_agent import SelfPlayCodeAgent
from evaluate import evaluate_checkpoint


def run_experiment():
    parser = argparse.ArgumentParser(description="Strict Multi-Phase Catastrophic Forgetting Experiment (True Self Play)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--run-name", type=str, default="strict_forgetting_eval")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--phase_steps", type=int, default=200, help="Steps per phase")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--full_finetune", action="store_true", help="Use Full Adaptation instead of LoRA")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train/eval datasets")
    # Skip Phase 0 and inject known baseline values directly (useful after a crash recovery)
    parser.add_argument("--base_eval_A", type=float, default=None,
                        help="Known baseline pass@1 for eval_A (skips Phase 0 re-evaluation)")
    parser.add_argument("--base_eval_B", type=float, default=None,
                        help="Known baseline pass@1 for eval_B (skips Phase 0 re-evaluation)")
    args = parser.parse_args()

    use_lora = not args.full_finetune

    wandb.init(project="anchored_code_self_play", name=args.run_name,
               config={"beta": args.beta, "K": args.K, "G": args.G, "use_lora": use_lora, "model_name": args.model_name})

    print(f"Initializing SelfPlay Agent (use_lora={use_lora}) with {args.model_name}...")
    agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=use_lora)

    # --- PHASE 0: Baseline Evaluation ---
    if args.base_eval_A is not None and args.base_eval_B is not None:
        print(f"\n--- PHASE 0: Skipping baseline (using provided values) ---")
        print(f"  base_eval_A = {args.base_eval_A:.4f}  base_eval_B = {args.base_eval_B:.4f}")
        base_eval_A = args.base_eval_A
        base_eval_B = args.base_eval_B
    else:
        print("\n--- PHASE 0: Baseline Evaluation ---")
        base_eval_A = evaluate_checkpoint("base", f"{args.data_dir}/eval_A.json", agent)
        base_eval_B = evaluate_checkpoint("base", f"{args.data_dir}/eval_B.json", agent)

    # Log baseline values at step 0 so they appear as horizontal reference lines in WandB
    wandb.log({
        "baseline/eval_A_pass_rate": base_eval_A,
        "baseline/eval_B_pass_rate": base_eval_B,
    }, step=0)

    # These closures capture base_eval_A/B to keep them constant across all steps
    def log_metrics(step, phase_name, train_metrics):
        wandb.log({
            f"{phase_name}/coder_loss": train_metrics.get("coder_loss", 0),
            f"{phase_name}/tester_loss": train_metrics.get("tester_loss", 0),
            f"{phase_name}/coder_success_rate": train_metrics.get("coder_success_rate", 0),
            f"{phase_name}/tester_success_rate": train_metrics.get("tester_success_rate", 0),
            "baseline/eval_A_pass_rate": base_eval_A,
            "baseline/eval_B_pass_rate": base_eval_B,
        }, step=step)

    def run_evals(step, checkpoint_path):
        eval_A = evaluate_checkpoint(checkpoint_path, f"{args.data_dir}/eval_A.json", agent)
        eval_B = evaluate_checkpoint(checkpoint_path, f"{args.data_dir}/eval_B.json", agent)
        wandb.log({
            "eval/eval_A_pass_rate": eval_A,
            "eval/eval_B_pass_rate": eval_B,
            "baseline/eval_A_pass_rate": base_eval_A,
            "baseline/eval_B_pass_rate": base_eval_B,
        }, step=step)

    # --- PHASE 1: Training on Distribution A ---
    print(f"\n--- PHASE 1: Training on Distribution A ({args.phase_steps} steps) ---")
    loop_A = CoderTesterSelfPlayLoop(agent=agent, dataset_path=f"{args.data_dir}/train_A.json", lr=1e-5, beta=args.beta)

    for step in range(1, args.phase_steps + 1):
        metrics = loop_A.run_step(G=args.G, K=args.K)
        if metrics:
            log_metrics(step, "Phase_A", metrics)

        if step % args.eval_freq == 0:
            ckpt_path = f"{args.ckpt_dir}/phaseA_step_{step}_self_play"
            agent.model.save_pretrained(ckpt_path)
            run_evals(step, ckpt_path)

    # --- PHASE 2: Training on Distribution B ---
    print(f"\n--- PHASE 2: Training on Distribution B ({args.phase_steps} steps) ---")
    loop_B = CoderTesterSelfPlayLoop(agent=agent, dataset_path=f"{args.data_dir}/train_B.json", lr=1e-5, beta=args.beta)

    for step in range(args.phase_steps + 1, (args.phase_steps * 2) + 1):
        metrics = loop_B.run_step(G=args.G, K=args.K)
        if metrics:
            log_metrics(step, "Phase_B", metrics)

        if step % args.eval_freq == 0:
            ckpt_path = f"{args.ckpt_dir}/phaseB_step_{step}_self_play"
            agent.model.save_pretrained(ckpt_path)
            run_evals(step, ckpt_path)

    print("\n--- EXPERIMENT COMPLETE ---")


if __name__ == "__main__":
    run_experiment()

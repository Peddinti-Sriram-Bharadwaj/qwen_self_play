import subprocess
import shutil
import os

def clean_checkpoints(directory):
    if os.path.exists(directory):
        print(f"Cleaning {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def run_ternary_search(low_lr=1e-5, high_lr=1e-4, max_iters=5, fixed_beta=0.01):
    print(f"Starting Parallel Ternary Search for Learning Rate threshold between {low_lr} and {high_lr} (Beta={fixed_beta})")
    
    for i in range(max_iters):
        # We interpolate linearly between the boundaries
        lr_1 = low_lr + (high_lr - low_lr) / 3.0
        lr_2 = low_lr + 2.0 * (high_lr - low_lr) / 3.0
        
        print("\n" + "="*50)
        print(f"ITERATION {i+1}/{max_iters}")
        print(f"Testing LR 1 = {lr_1:.7f} on GPU 0")
        print(f"Testing LR 2 = {lr_2:.7f} on GPU 1")
        print("="*50)
        
        ckpt_dir_1 = "checkpoints_sweep_lr_0"
        ckpt_dir_2 = "checkpoints_sweep_lr_1"
        clean_checkpoints(ckpt_dir_1)
        clean_checkpoints(ckpt_dir_2)
        
        # 1. Run main experiment concurrently on both GPUs
        cmd_train_1 = [
            "python", "-u", "main_experiment.py",
            "--gpu", "0",
            "--beta", str(fixed_beta),
            "--lr", str(lr_1),
            "--phase_steps", "100",
            "--eval_freq", "100",
            "--ckpt_dir", ckpt_dir_1,
            "--skip_phase2",
            "--full_finetune",
            "--run-name", f"ternary_search_lr_{lr_1:.7f}",
            "--base_eval_A", "0.0",
            "--base_eval_B", "0.0",
            "--model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        ]
        
        cmd_train_2 = [
            "python", "-u", "main_experiment.py",
            "--gpu", "1",
            "--beta", str(fixed_beta),
            "--lr", str(lr_2),
            "--phase_steps", "100",
            "--eval_freq", "100",
            "--ckpt_dir", ckpt_dir_2,
            "--skip_phase2",
            "--full_finetune",
            "--run-name", f"ternary_search_lr_{lr_2:.7f}",
            "--base_eval_A", "0.0",
            "--base_eval_B", "0.0",
            "--model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        ]
        
        print("Running Phase 1 Training concurrently on both GPUs...")
        p1 = subprocess.Popen(cmd_train_1)
        p2 = subprocess.Popen(cmd_train_2)
        
        p1.wait()
        p2.wait()
        
        if p1.returncode != 0 or p2.returncode != 0:
            print("Training failed on one or both GPUs! Aborting search.")
            break
            
        # 2. Run Conjugate Eval sequentially (to save CPU/RAM)
        print("\nRunning Conjugate Prompting Evaluations...")
        cmd_eval_1 = [
            "python", "-u", "conjugate_prompting_eval.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_1,
            "--headless", "--skip_baseline"
        ]
        cmd_eval_2 = [
            "python", "-u", "conjugate_prompting_eval.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_2,
            "--headless", "--skip_baseline"
        ]
        
        res_1 = subprocess.run(cmd_eval_1)
        res_2 = subprocess.run(cmd_eval_2)
        
        rec_1 = (res_1.returncode == 0)
        rec_2 = (res_2.returncode == 0)
        
        print(f"--> [RESULT] LR 1 ({lr_1:.7f}) : {'RECOVERED' if rec_1 else 'COLLAPSED'}")
        print(f"--> [RESULT] LR 2 ({lr_2:.7f}) : {'RECOVERED' if rec_2 else 'COLLAPSED'}")
        
        # 3. Decision Logic
        # A HIGHER learning rate causes COLLAPSE. A LOWER learning rate causes RECOVERY.
        if rec_1 and rec_2:
            # Both recovered, so the true collapse threshold must be higher than lr_2
            low_lr = lr_2
        elif not rec_1 and not rec_2:
            # Both collapsed, so the true collapse threshold must be lower than lr_1
            high_lr = lr_1
        else:
            # lr_1 recovered, lr_2 collapsed (perfect sandwich)
            low_lr = lr_1
            high_lr = lr_2
            
        # 4. Run Sahoo Diagnostics
        print("\nRunning Sahoo Unified Diagnostics for both...")
        subprocess.run([
            "python", "-u", "sahoo_unified_diagnostics.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_1,
            "--run_label", f"LR_{lr_1:.7f}"
        ])
        subprocess.run([
            "python", "-u", "sahoo_unified_diagnostics.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_2,
            "--run_label", f"LR_{lr_2:.7f}"
        ])
        
        print(f"\nNew Search Range: [{low_lr:.7f}, {high_lr:.7f}]")
        
    print("\n" + "="*50)
    print(f"PARALLEL TERNARY SEARCH COMPLETE.")
    print(f"The phase transition threshold for True Representation Collapse is approx LR: {(low_lr + high_lr) / 2:.7f}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel Ternary Search for Learning Rate Phase Transition")
    parser.add_argument("--low_lr", type=float, default=1e-5)
    parser.add_argument("--high_lr", type=float, default=1e-4)
    parser.add_argument("--max_iters", type=int, default=5)
    parser.add_argument("--fixed_beta", type=float, default=0.01)
    args = parser.parse_args()
    
    run_ternary_search(low_lr=args.low_lr, high_lr=args.high_lr, max_iters=args.max_iters, fixed_beta=args.fixed_beta)

import subprocess
import shutil
import os

def clean_checkpoints(directory):
    if os.path.exists(directory):
        print(f"Cleaning {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def run_binary_search(low_beta=0.001, high_beta=0.04, max_iters=6):
    print(f"Starting Binary Search for KL Beta threshold between {low_beta} and {high_beta}")
    
    for i in range(max_iters):
        mid = (low_beta + high_beta) / 2
        print("\n" + "="*50)
        print(f"ITERATION {i+1}/{max_iters} | Testing Beta = {mid:.5f}")
        print("="*50)
        
        ckpt_dir = "checkpoints_sweep"
        clean_checkpoints(ckpt_dir)
        
        # 1. Run main experiment for 100 steps
        cmd_train = [
            "python", "-u", "main_experiment.py",
            "--gpu", "0",  # Default to GPU 0, user can override with CUDA_VISIBLE_DEVICES
            "--beta", str(mid),
            "--phase_steps", "100",
            "--eval_freq", "100",
            "--ckpt_dir", ckpt_dir,
            "--skip_phase2",
            "--run-name", f"binary_search_beta_{mid:.5f}",
            "--base_eval_A", "0.0", # Skip phase 0 evaluation
            "--base_eval_B", "0.0",
            "--model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        ]
        
        print(f"Running Phase 1 Training (100 steps)...")
        # Ensure it doesn't crash the orchestrator if training fails for some reason
        train_result = subprocess.run(cmd_train)
        if train_result.returncode != 0:
            print("Training failed! Aborting binary search.")
            break
        
        # 2. Run Conjugate Eval
        cmd_eval = [
            "python", "-u", "conjugate_prompting_eval.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir,
            "--headless",
            "--skip_baseline"
        ]
        
        print("Running Conjugate Prompting Evaluation...")
        result = subprocess.run(cmd_eval)
        
        if result.returncode == 0:
            print(f"--> [RESULT] Beta {mid:.5f} RECOVERED (Mere Suppression).")
            # This beta is high enough to protect weights. The true collapse threshold is lower.
            high_beta = mid
        else:
            print(f"--> [RESULT] Beta {mid:.5f} COLLAPSED (True Destruction).")
            # This beta is too low, weights got destroyed. True collapse threshold is higher.
            low_beta = mid
            
        # 3. Run Sahoo Unified Diagnostics to track CKA and Alignment Drift
        cmd_sahoo = [
            "python", "-u", "sahoo_unified_diagnostics.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir
        ]
        print("\nRunning Sahoo Unified Diagnostics (CKA, GDI, Alignment Drift)...")
        subprocess.run(cmd_sahoo)
            
        print(f"\nNew Search Range: [{low_beta:.5f}, {high_beta:.5f}]")
        
    print("\n" + "="*50)
    print(f"BINARY SEARCH COMPLETE.")
    print(f"The phase transition threshold for True Representation Collapse is approximately: {(low_beta + high_beta) / 2:.5f}")
    print("="*50)

if __name__ == "__main__":
    run_binary_search()

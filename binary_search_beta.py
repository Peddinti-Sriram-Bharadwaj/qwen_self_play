import subprocess
import shutil
import os

def clean_checkpoints(directory):
    if os.path.exists(directory):
        print(f"Cleaning {directory}...")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def run_ternary_search(low_beta=0.001, high_beta=0.04, max_iters=5):
    print(f"Starting Parallel Ternary Search for KL Beta threshold between {low_beta} and {high_beta}")
    
    for i in range(max_iters):
        # Calculate 1/3 and 2/3 points
        beta_1 = low_beta + (high_beta - low_beta) / 3.0
        beta_2 = low_beta + 2.0 * (high_beta - low_beta) / 3.0
        
        print("\n" + "="*50)
        print(f"ITERATION {i+1}/{max_iters}")
        print(f"Testing Beta 1 = {beta_1:.5f} on GPU 0")
        print(f"Testing Beta 2 = {beta_2:.5f} on GPU 1")
        print("="*50)
        
        ckpt_dir_1 = "checkpoints_sweep_0"
        ckpt_dir_2 = "checkpoints_sweep_1"
        clean_checkpoints(ckpt_dir_1)
        clean_checkpoints(ckpt_dir_2)
        
        # 1. Run main experiment concurrently on both GPUs
        cmd_train_1 = [
            "python", "-u", "main_experiment.py",
            "--gpu", "0",
            "--beta", str(beta_1),
            "--phase_steps", "100",
            "--eval_freq", "100",
            "--ckpt_dir", ckpt_dir_1,
            "--skip_phase2",
            "--run-name", f"ternary_search_beta_{beta_1:.5f}",
            "--base_eval_A", "0.0",
            "--base_eval_B", "0.0",
            "--model_name", "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        ]
        
        cmd_train_2 = [
            "python", "-u", "main_experiment.py",
            "--gpu", "1",
            "--beta", str(beta_2),
            "--phase_steps", "100",
            "--eval_freq", "100",
            "--ckpt_dir", ckpt_dir_2,
            "--skip_phase2",
            "--run-name", f"ternary_search_beta_{beta_2:.5f}",
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
            
        # 2. Run Conjugate Eval sequentially (or concurrently, but sequential is safer for CPU/RAM)
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
        
        # We evaluate sequentially to avoid out of memory errors when loading the LLM multiple times
        res_1 = subprocess.run(cmd_eval_1)
        res_2 = subprocess.run(cmd_eval_2)
        
        rec_1 = (res_1.returncode == 0)
        rec_2 = (res_2.returncode == 0)
        
        print(f"--> [RESULT] Beta 1 ({beta_1:.5f}) : {'RECOVERED' if rec_1 else 'COLLAPSED'}")
        print(f"--> [RESULT] Beta 2 ({beta_2:.5f}) : {'RECOVERED' if rec_2 else 'COLLAPSED'}")
        
        # 3. Decision Logic
        if rec_1 and rec_2:
            # Both recovered, threshold is lower than beta_1
            high_beta = beta_1
        elif not rec_1 and not rec_2:
            # Both collapsed, threshold is higher than beta_2
            low_beta = beta_2
        else:
            # beta_1 collapsed, beta_2 recovered (perfect sandwich)
            low_beta = beta_1
            high_beta = beta_2
            
        # 4. Run Sahoo Diagnostics
        print("\nRunning Sahoo Unified Diagnostics for both...")
        subprocess.run([
            "python", "-u", "sahoo_unified_diagnostics.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_1
        ])
        subprocess.run([
            "python", "-u", "sahoo_unified_diagnostics.py",
            "--base_model", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "--rl_checkpoints_dir", ckpt_dir_2
        ])
        
        print(f"\nNew Search Range: [{low_beta:.5f}, {high_beta:.5f}]")
        
    print("\n" + "="*50)
    print(f"PARALLEL TERNARY SEARCH COMPLETE.")
    print(f"The phase transition threshold for True Representation Collapse is approximately: {(low_beta + high_beta) / 2:.5f}")
    print("="*50)

if __name__ == "__main__":
    run_ternary_search()

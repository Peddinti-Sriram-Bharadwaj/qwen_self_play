import os
import argparse
import torch
import gc
import json
import matplotlib.pyplot as plt

# Isolate GPU before loading torch/transformers
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_lora")
parser.add_argument("--output_prefix", type=str, default="lora")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from code_agent import SelfPlayCodeAgent
from diagnostics.target_distillation import generate_distillation_datasets, run_target_distillation
from transformers import AutoModelForCausalLM

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Longitudinal Diagnostics on {device}")
    
    # 1. Initialize Base Model & Datasets
    print("Loading Base Model...")
    base_agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=True)
    datasets = generate_distillation_datasets(base_agent.tokenizer)
    
    results = {}
    
    # Evaluate Base Model (Step 0)
    print("\n--- Evaluating Base Model (Step 0) ---")
    base_agent.model.eval()
    
    # Disable adapters to evaluate raw base model
    with base_agent.model.disable_adapter():
        hard_loss = run_target_distillation(base_agent.model, datasets["hard"], device, steps=50)
        relearn_loss = run_target_distillation(base_agent.model, datasets["relearn"], device, steps=50)
        
    results[0] = {
        "plasticity_auc": sum(hard_loss),
        "plasticity_final": hard_loss[-1],
        "relearn_auc": sum(relearn_loss),
        "relearn_final": relearn_loss[-1]
    }
    
    # 2. Find Checkpoints
    ckpts = sorted(
        [d for d in os.listdir(args.checkpoints_dir)
         if os.path.isdir(os.path.join(args.checkpoints_dir, d)) 
         and "self_play" in d 
         and ("phaseA" in d or "phaseB" in d)],
        key=lambda x: int(x.split("step_")[1].split("_")[0]) if "step_" in x else 0
    )
    
    if not ckpts:
        print(f"No checkpoints found in {args.checkpoints_dir}")
        return
        
    for ckpt_name in ckpts:
        ckpt_path = os.path.join(args.checkpoints_dir, ckpt_name)
        step = int(ckpt_name.split("step_")[1].split("_")[0])
        print(f"\n--- Evaluating Checkpoint: {ckpt_name} (Step {step}) ---")
        
        adapter_path = os.path.join(ckpt_path, "self_play")
        is_peft = os.path.exists(os.path.join(ckpt_path, "adapter_model.safetensors")) or \
                  os.path.exists(os.path.join(ckpt_path, "adapter_model.bin")) or \
                  os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")) or \
                  os.path.exists(os.path.join(adapter_path, "adapter_model.bin"))
                  
        if is_peft:
            actual_adapter_dir = adapter_path if os.path.exists(adapter_path) else ckpt_path
            # Reload base weights to undo previous distillation gradients!
            base_agent.model.unload()
            base_agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=True)
            base_agent.model.load_adapter(actual_adapter_dir, adapter_name="self_play")
            
            hard_loss = run_target_distillation(base_agent.model, datasets["hard"], device, steps=50)
            
            # Reload again before second distillation so gradients don't leak
            base_agent.model.unload()
            base_agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=True)
            base_agent.model.load_adapter(actual_adapter_dir, adapter_name="self_play")
            
            relearn_loss = run_target_distillation(base_agent.model, datasets["relearn"], device, steps=50)
            
        else:
            # Full Model
            full_agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=False)
            full_agent.model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to(device)
            hard_loss = run_target_distillation(full_agent.model, datasets["hard"], device, steps=50)
            del full_agent
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reload for Relearn
            full_agent = SelfPlayCodeAgent(model_name=args.model_name, use_lora=False)
            full_agent.model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16).to(device)
            relearn_loss = run_target_distillation(full_agent.model, datasets["relearn"], device, steps=50)
            del full_agent
            gc.collect()
            torch.cuda.empty_cache()
            
        results[step] = {
            "plasticity_auc": sum(hard_loss),
            "plasticity_final": hard_loss[-1],
            "relearn_auc": sum(relearn_loss),
            "relearn_final": relearn_loss[-1]
        }
        
    # Save raw results
    with open(f"{args.output_prefix}_longitudinal_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Plotting
    steps = sorted(results.keys())
    plasticity_auc = [results[s]["plasticity_auc"] for s in steps]
    relearn_auc = [results[s]["relearn_auc"] for s in steps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(steps, plasticity_auc, marker='o', color='red', label='Plasticity AUC (Hard Task)')
    ax1.set_title("Adaptation Efficiency (Plasticity)\nLower is better")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss AUC")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2.plot(steps, relearn_auc, marker='o', color='blue', label='Relearning AUC (Task A)')
    ax2.set_title("Retention & Recovery (Forgetting)\nLower is better")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss AUC")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle(f"Longitudinal Trajectories: {args.output_prefix.upper()} Fine-Tuning")
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_longitudinal_plot.png", dpi=300)
    print(f"\nSaved plots to {args.output_prefix}_longitudinal_plot.png")

if __name__ == "__main__":
    main()

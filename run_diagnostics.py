import argparse
import os

# Parse the GPU argument BEFORE importing PyTorch to guarantee strict VRAM isolation
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0", help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES)")
_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _args.gpu

import torch
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from diagnostics.target_distillation import generate_distillation_datasets, run_target_distillation
from diagnostics.activation_sparsity import measure_dormant_neurons
from diagnostics.effective_rank import measure_effective_rank
from diagnostics.cka import measure_representational_warp

def load_base_model(model_name, device):
    print(f"Loading Base Model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    return model, tokenizer

def load_lora_model(base_model, lora_path):
    print(f"Loading LoRA Model from {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    return lora_model

def load_full_model(full_path, device):
    print(f"Loading Full-Trained Model from {full_path}")
    model = AutoModelForCausalLM.from_pretrained(full_path, torch_dtype=torch.bfloat16).to(device)
    return model

def plot_loss_curves(base_losses, lora_losses, full_losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(base_losses, label="Base Model", color="blue")
    plt.plot(lora_losses, label="LoRA Model", color="orange")
    plt.plot(full_losses, label="Full-Trained Model", color="red")
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def run_diagnostics():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--lora_ckpt", type=str, default="checkpoints_lora/phaseB_step_400_self_play")
    parser.add_argument("--full_ckpt", type=str, default="checkpoints_full/phaseB_step_400_self_play")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    os.makedirs("diagnostic_results", exist_ok=True)
    
    # 1. Load Models
    base_model, tokenizer = load_base_model(args.base_model, args.device)
    
    # Generate common datasets
    distillation_datasets = generate_distillation_datasets(tokenizer)
    validation_dataset = distillation_datasets["medium"] # Use a small MBPP batch for static analysis
    
    print("\n==================================================")
    print(" METHOD 2 & 3: Dormancy & Effective Rank (BASE)")
    print("==================================================")
    base_dormancy = measure_dormant_neurons(base_model, validation_dataset, args.device)
    base_rank = measure_effective_rank(base_model, validation_dataset, args.device)
    print(f"Base Model Effective Rank: {base_rank:.2f}")
    
    print("\n==================================================")
    print(" METHOD 1: Control Target Distillation (BASE)")
    print("==================================================")
    # We must clone or reload the model for distillation since it alters weights
    base_clone = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(args.device)
    base_loss_easy = run_target_distillation(base_clone, distillation_datasets["easy"], args.device)
    base_loss_hard = run_target_distillation(base_clone, distillation_datasets["hard"], args.device)
    del base_clone
    
    # Now load LoRA over the base model
    lora_model = load_lora_model(base_model, args.lora_ckpt)
    
    print("\n==================================================")
    print(" METHOD 2 & 3 & 4: Diagnostics (LORA)")
    print("==================================================")
    # Disable adapter temporarily for CKA base reference
    with lora_model.disable_adapter():
        base_ref = lora_model
    
    lora_cka = measure_representational_warp(base_model, lora_model, validation_dataset, args.device)
    lora_dormancy = measure_dormant_neurons(lora_model, validation_dataset, args.device)
    lora_rank = measure_effective_rank(lora_model, validation_dataset, args.device)
    print(f"LoRA Model Effective Rank: {lora_rank:.2f}")
    
    print("\n==================================================")
    print(" METHOD 1: Control Target Distillation (LORA)")
    print("==================================================")
    lora_loss_easy = run_target_distillation(lora_model, distillation_datasets["easy"], args.device)
    lora_loss_hard = run_target_distillation(lora_model, distillation_datasets["hard"], args.device)
    
    del lora_model, base_model
    torch.cuda.empty_cache()
    
    print("\n==================================================")
    print(" METHOD 2 & 3 & 4: Diagnostics (FULL)")
    print("==================================================")
    full_model = load_full_model(args.full_ckpt, args.device)
    # Load base again for CKA reference
    base_ref = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(args.device)
    
    full_cka = measure_representational_warp(base_ref, full_model, validation_dataset, args.device)
    full_dormancy = measure_dormant_neurons(full_model, validation_dataset, args.device)
    full_rank = measure_effective_rank(full_model, validation_dataset, args.device)
    print(f"Full Model Effective Rank: {full_rank:.2f}")
    
    print("\n==================================================")
    print(" METHOD 1: Control Target Distillation (FULL)")
    print("==================================================")
    full_loss_easy = run_target_distillation(full_model, distillation_datasets["easy"], args.device)
    full_loss_hard = run_target_distillation(full_model, distillation_datasets["hard"], args.device)
    
    print("\n==================================================")
    print(" SAVING DIAGNOSTIC RESULTS")
    print("==================================================")
    plot_loss_curves(base_loss_easy, lora_loss_easy, full_loss_easy, "Distillation (Easy Task)", "diagnostic_results/distillation_easy.png")
    plot_loss_curves(base_loss_hard, lora_loss_hard, full_loss_hard, "Distillation (Hard Task)", "diagnostic_results/distillation_hard.png")
    
    report = f"""# Plasticity Diagnostics Report

## Method 1: Control Target Distillation
Loss curves have been saved to `diagnostic_results/`. If LoRA or Full models plateau higher than Base, they have lost plasticity.

## Method 2: Activation Sparsity (Average Dormancy)
- Base Model: {sum(base_dormancy.values())/len(base_dormancy):.2f}% dormant
- LoRA Model: {sum(lora_dormancy.values())/len(lora_dormancy):.2f}% dormant
- Full Model: {sum(full_dormancy.values())/len(full_dormancy):.2f}% dormant

## Method 3: Feature Expressivity (Effective Rank)
- Base Model Rank: {base_rank:.2f}
- LoRA Model Rank: {lora_rank:.2f}
- Full Model Rank: {full_rank:.2f}
*(Drop in rank = representation collapse)*

## Method 4: Representational Warp (Average CKA)
- LoRA vs Base CKA: {sum(lora_cka.values())/len(lora_cka):.4f}
- Full vs Base CKA: {sum(full_cka.values())/len(full_cka):.4f}
*(CKA < 0.9 = severe geometric warping)*
"""
    with open("diagnostic_results/plasticity_report.md", "w") as f:
        f.write(report)
        
    print("Diagnostics complete! See diagnostic_results/plasticity_report.md")

if __name__ == "__main__":
    run_diagnostics()

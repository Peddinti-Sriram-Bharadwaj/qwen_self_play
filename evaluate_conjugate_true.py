import json
import argparse
import os
import glob
import torch

from code_agent import SelfPlayCodeAgent
from parsing import extract_python_code
from sandbox import evaluate_code
from prompts import FIX_PROMPT, SYSTEM_MSG
from conjugate_prompting_eval import TRANSFORMS

def run_true_conjugate_eval(checkpoints_dir, dataset_path):
    print(f"\n[{'='*60}]")
    print(f" TRUE CONJUGATE PROMPTING EVALUATION (SANDBOX EXEC)")
    print(f"[{'='*60}]\n")
    print(f"Checkpoints Directory: {checkpoints_dir}")
    print(f"Dataset: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    target_checkpoints = []
    for ckpt in sorted(glob.glob(os.path.join(checkpoints_dir, "phase*_step_*"))):
        if os.path.isdir(ckpt):
            target_checkpoints.append(ckpt)
            
    if not target_checkpoints:
        print(f"No checkpoint folders matching 'phase*_step_*' found in {checkpoints_dir}!")
        return
        
    total = len(dataset)
    all_results = {}

    for checkpoint_path in target_checkpoints:
        step_name = os.path.basename(checkpoint_path)
        print(f"\n\n{'*'*60}")
        print(f" EVALUATING CHECKPOINT: {step_name}")
        print(f"{'*'*60}\n")
        
        print(f"Loading Model from {checkpoint_path}...")
        
        # Check if it's a LoRA adapter or a Full Model
        is_peft = os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
                  os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin"))
                  
        if is_peft:
            print(f"  -> Detected LoRA adapter. Loading base model Qwen/Qwen2.5-Coder-0.5B-Instruct first...")
            agent = SelfPlayCodeAgent(model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct", use_lora=True)
            agent.model.load_adapter(checkpoint_path, adapter_name="self_play")
        else:
            print(f"  -> Detected Full Model checkpoint.")
            agent = SelfPlayCodeAgent(model_name=checkpoint_path, use_lora=False)
        
        results = {}

        for t_name, t_config in TRANSFORMS.items():
            print(f"\n--- Running Transform: {t_name} ---")
            passed = 0
            
            for i, task in enumerate(dataset):
                tests_str = "\n".join(task["tests"])
                
                # Translate ONLY the natural language problem description
                transformed_problem = t_config["fwd"](task["problem"])
                
                raw_prompt = FIX_PROMPT.format(
                    problem=transformed_problem,
                    tests=tests_str,
                    buggy_code=task["buggy_code"]
                )

                messages = [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": raw_prompt}
                ]
                prompt = agent.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                raw_fixes = agent.batched_generate(
                    [prompt], 
                    max_tokens=512, 
                    temperature=0.01,
                )
                
                parsed_fix = extract_python_code(raw_fixes[0])
                res = evaluate_code(parsed_fix, task["tests"])

                if res.all_passed:
                    passed += 1

                print(f"[{i+1}/{total}] Task {task['task_id']} | {t_name} Passed: {res.all_passed} | Current pass@1: {(passed/(i+1))*100:.2f}%")
                
            final_pass = passed / total
            print(f"\nFINAL pass@1 for {t_name}: {final_pass*100:.2f}%")
            results[t_name] = final_pass
            
        all_results[step_name] = results
        
        # Clean up memory before loading the next checkpoint
        del agent
        torch.cuda.empty_cache()

    print("\n\n" + "="*70)
    print(" TRUE SANDBOX PASS RATE METRICS (ALL CHECKPOINTS)")
    print("="*70)
    
    # Print a nice summary table at the end
    print(f"{'Checkpoint':<25} | ", end="")
    for t_name in TRANSFORMS.keys():
        print(f"{t_name:<12} | ", end="")
    print("\n" + "-"*70)
    
    for ckpt_name, res in all_results.items():
        print(f"{ckpt_name:<25} | ", end="")
        for t_name in TRANSFORMS.keys():
            score = res.get(t_name, 0.0)
            print(f"{score*100:6.2f}%      | ", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=True, help="Directory containing the phase*_step_* checkpoints")
    parser.add_argument("--dataset", type=str, default="data/eval_A.json", help="Path to evaluation dataset")
    args = parser.parse_args()

    run_true_conjugate_eval(args.checkpoints_dir, args.dataset)

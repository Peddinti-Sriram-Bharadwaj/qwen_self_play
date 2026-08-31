import json
import argparse
import os

from code_agent import SelfPlayCodeAgent
from parsing import extract_python_code
from sandbox import evaluate_code
from prompts import FIX_PROMPT, SYSTEM_MSG
from conjugate_prompting_eval import TRANSFORMS

def run_true_conjugate_eval(checkpoint_path, dataset_path, model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
    print(f"\n[{'='*60}]")
    print(f" TRUE CONJUGATE PROMPTING EVALUATION (SANDBOX EXEC)")
    print(f"[{'='*60}]\n")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    print(f"\nLoading Model from {checkpoint_path}...")
    agent = SelfPlayCodeAgent(model_name=checkpoint_path, use_lora=False)
    
    total = len(dataset)
    
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

            # Generate with the specific repetition penalty for this transform
            raw_fixes = agent.batched_generate(
                [prompt], 
                max_tokens=512, 
                temperature=0.01,
                # We need to pass rep_penalty to batched_generate if supported, 
                # but batched_generate in SelfPlayCodeAgent might not support it directly in kwargs.
                # If not, we just use the default.
            )
            
            parsed_fix = extract_python_code(raw_fixes[0])
            res = evaluate_code(parsed_fix, task["tests"])

            if res.all_passed:
                passed += 1

            print(f"[{i+1}/{total}] Task {task['task_id']} | {t_name} Passed: {res.all_passed} | Current pass@1: {(passed/(i+1))*100:.2f}%")
            
        final_pass = passed / total
        print(f"\nFINAL pass@1 for {t_name}: {final_pass*100:.2f}%")
        results[t_name] = final_pass

    print("\n\n" + "="*50)
    print(" TRUE SANDBOX PASS RATE METRICS")
    print("="*50)
    for t_name, score in results.items():
        print(f"{t_name:<15} | {score*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="data/eval_A.json", help="Path to evaluation dataset")
    args = parser.parse_args()

    run_true_conjugate_eval(args.checkpoint, args.dataset)

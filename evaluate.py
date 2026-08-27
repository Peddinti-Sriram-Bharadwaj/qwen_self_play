import json
import torch
import numpy as np
from code_agent import DualLoraCodeAgent, extract_python_code
from sandbox import evaluate_code
import argparse

def evaluate_checkpoint(checkpoint_path, dataset_path, agent=None):
    print(f"\nEvaluating Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    # We always load the model with the 'fixer' role active since we are evaluating repair
    if agent is None:
        agent = DualLoraCodeAgent(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
        if checkpoint_path != "base":
            print(f"Loading LoRA weights from {checkpoint_path}...")
            agent.model.load_adapter(checkpoint_path, adapter_name="fixer")
    else:
        # If agent is provided, we assume it already has the correct weights loaded
        pass
        
    agent.set_active_role("fixer")
        
    fix_prompt_template = """You are an expert Python debugger. Here is a problem description, a suite of unit tests, and a buggy implementation.
Your task is to fix the bug so that all tests pass.
First, reason about the bug and how to fix it step-by-step inside a <think> block.
Then, output the corrected Python code block.

Problem: {problem}
Tests: {tests}

Buggy Code:
```python
{buggy_code}
```
"""

    passed = 0
    total = len(dataset)
    
    for i, task in enumerate(dataset):
        tests_str = "\n".join(task["tests"])
        prompt = fix_prompt_template.format(
            problem=task["problem"],
            tests=tests_str,
            buggy_code=task["buggy_code"]
        )
        
        # pass@1 strict evaluation (Temperature 0.0)
        # We generate 1 sample. Since greedy decoding isn't always supported smoothly in all batched implementations,
        # we can use temperature 0.01 to approximate it.
        raw_fixes = agent.batched_generate([prompt], adapter_name="fixer", max_tokens=256, temperature=0.01)
        parsed_fix = extract_python_code(raw_fixes[0])
        
        res = evaluate_code(parsed_fix, task["tests"])
        
        if res["all_passed"]:
            passed += 1
            
        print(f"[{i+1}/{total}] Task {task['task_id']} | Passed: {res['all_passed']} | Current pass@1: {(passed/(i+1))*100:.2f}%")
        
    final_pass_rate = passed / total
    print(f"\nFINAL pass@1 for {dataset_path}: {final_pass_rate*100:.2f}%")
    return final_pass_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="base", help="Path to checkpoint (or 'base')")
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset (e.g., data/eval_A.json)")
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.dataset)

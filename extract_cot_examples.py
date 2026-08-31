"""
Extract Chain-of-Thought (CoT) Examples from RL Checkpoints
===========================================================
This script iterates through the base model and all RL checkpoints,
prompts them with standard coding tasks, and logs their complete 
raw generation (including the <think> blocks) to a Markdown file.

This allows us to visually inspect how the CoT reasoning degrades 
or collapses (e.g. turning into gibberish) over the training process.
"""

import torch
import os
import glob
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import CODER_PROMPT

TASKS = [
    {
        "id": "sort_list",
        "problem": "Write a Python function to sort a list of integers. Name it `sort_list`.",
        "tests": "assert sort_list([3, 1, 2]) == [1, 2, 3]\nassert sort_list([]) == []"
    },
    {
        "id": "is_palindrome",
        "problem": "Write a Python function to check if a string is a palindrome. Name it `is_palindrome`.",
        "tests": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False"
    }
]

def generate_example(model, tokenizer, prompt, max_new_tokens=512):
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_len = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id
        )
        
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def run_extraction(base_model_path, rl_checkpoints_dir, output_file="cot_evolution_log.md"):
    print(f"Starting CoT extraction. Output will be saved to: {output_file}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(output_file, "w") as f:
        f.write("# Chain-of-Thought (CoT) Evolution Log\n\n")
        f.write("Tracking how the model's `<think>` reasoning traces change across RL checkpoints.\n\n")
        
    # Evaluate Base Model
    print("--- Evaluating Base Model ---")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    with open(output_file, "a") as f:
        f.write("## Checkpoint: BASE MODEL\n")
        for task in TASKS:
            prompt = CODER_PROMPT.format(problem=task["problem"], tests=task["tests"])
            print(f"  Generating for {task['id']}...")
            output = generate_example(base_model, tokenizer, prompt)
            f.write(f"### Task: {task['id']}\n```python\n{output}\n```\n\n")
            
    del base_model
    torch.cuda.empty_cache()
    
    # Evaluate Checkpoints
    checkpoints = sorted(glob.glob(os.path.join(rl_checkpoints_dir, "phase*_step_*")))
    
    for ckpt in checkpoints:
        step_name = os.path.basename(ckpt)
        print(f"--- Evaluating {step_name} ---")
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")
        
        with open(output_file, "a") as f:
            f.write(f"---\n## Checkpoint: {step_name}\n")
            for task in TASKS:
                prompt = CODER_PROMPT.format(problem=task["problem"], tests=task["tests"])
                print(f"  Generating for {task['id']}...")
                output = generate_example(model, tokenizer, prompt)
                f.write(f"### Task: {task['id']}\n```python\n{output}\n```\n\n")
                
        del model
        torch.cuda.empty_cache()
        
    print(f"Done! Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--rl_checkpoints_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="cot_evolution_log.md")
    args = parser.parse_args()
    
    run_extraction(args.base_model, args.rl_checkpoints_dir, args.output_file)

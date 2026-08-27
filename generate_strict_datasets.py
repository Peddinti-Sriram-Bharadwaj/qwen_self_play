import json
import subprocess
import sys
import os
import random

def ensure_datasets_installed():
    try:
        import datasets
    except ImportError:
        print("HuggingFace 'datasets' library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def categorize_task(prompt):
    prompt = prompt.lower()
    dist_a_keywords = ['string', 'list', 'dict', 'regex', 'sort', 'word', 'char', 'text', 'tuple']
    dist_b_keywords = ['math', 'num', 'calc', 'sum', 'mul', 'div', 'int', 'eq', 'mat', 'geo', 'graph', 'tree', 'array']
    
    a_score = sum(1 for k in dist_a_keywords if k in prompt)
    b_score = sum(1 for k in dist_b_keywords if k in prompt)
    
    if a_score > b_score:
        return 'A'
    elif b_score > a_score:
        return 'B'
    else:
        return random.choice(['A', 'B']) # Tie breaker

def generate_datasets():
    ensure_datasets_installed()
    from datasets import load_dataset
    
    os.makedirs("data", exist_ok=True)
    
    print("Downloading MBPP dataset for Training Distributions...")
    mbpp = load_dataset("google-research-datasets/mbpp", "sanitized")
    
    train_A = []
    train_B = []
    
    for item in mbpp["train"]:
        cat = categorize_task(item['prompt'])
        task = {
            "task_id": f"mbpp_{item['task_id']}",
            "problem": item['prompt'],
            "correct_code": item['code'],
            "tests": item['test_list']
        }
        if cat == 'A' and len(train_A) < 400:
            train_A.append(task)
        elif cat == 'B' and len(train_B) < 400:
            train_B.append(task)
            
    print(f"Generated MBPP Train A: {len(train_A)} tasks.")
    print(f"Generated MBPP Train B: {len(train_B)} tasks.")
    
    with open("data/train_A.json", "w") as f:
        json.dump(train_A, f, indent=4)
    with open("data/train_B.json", "w") as f:
        json.dump(train_B, f, indent=4)
        
    print("\nDownloading bigcode/humanevalpack for Strict Evaluations...")
    hep = load_dataset("bigcode/humanevalpack", "python")
    
    eval_A = []
    eval_B = []
    
    for item in hep["test"]:
        cat = categorize_task(item['instruction'])
        # Humanevalpack format: instruction, buggy_solution, test
        task = {
            "task_id": f"hep_{item['task_id']}",
            "problem": item['instruction'],
            "buggy_code": item['buggy_solution'],
            "tests": [item['test']] # It usually provides a single large test string block
        }
        if cat == 'A':
            eval_A.append(task)
        else:
            eval_B.append(task)
            
    print(f"Generated HumanEvalFix Eval A: {len(eval_A)} bugs.")
    print(f"Generated HumanEvalFix Eval B: {len(eval_B)} bugs.")
    
    with open("data/eval_A.json", "w") as f:
        json.dump(eval_A, f, indent=4)
    with open("data/eval_B.json", "w") as f:
        json.dump(eval_B, f, indent=4)
        
    print("\nAll strict datasets successfully generated in data/ directory!")

if __name__ == "__main__":
    generate_datasets()

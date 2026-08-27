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
        
    print("\nGenerating synthetic sentences for NLP Domain Conflict (Phase B)...")
    base_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Artificial intelligence is transforming the software industry rapidly.",
        "The committee decided to postpone the meeting until next week.",
        "Despite the heavy rain, the football match continued as planned.",
        "I have never been to Paris, but I hope to visit it someday.",
        "The recipe requires two cups of flour and one cup of sugar.",
        "He opened the door cautiously, expecting to find someone there.",
        "Learning a new programming language can be both challenging and rewarding.",
        "The sun set behind the mountains, casting a golden glow over the valley."
    ]
    
    train_B_nlp = []
    for i in range(400):
        # Repeat the sentences to create 400 tasks, appending a unique id/number to ensure variety
        text = base_sentences[i % len(base_sentences)]
        train_B_nlp.append({
            "task_id": f"nlp_synth_{i}",
            "correct_text": f"{text} This is instance number {i}."
        })
            
    print(f"Generated Synthetic Train B (NLP): {len(train_B_nlp)} sentences.")
    with open("data/train_B_nlp.json", "w") as f:
        json.dump(train_B_nlp, f, indent=4)
        
    print("\nDownloading bigcode/humanevalpack for Strict Evaluations...")
    hep = load_dataset("bigcode/humanevalpack", "python")
    
    eval_A = []
    eval_B = []
    
    for item in hep["test"]:
        cat = categorize_task(item['instruction'])
        # Humanevalpack format
        imports = item.get('import', '')
        declaration = item.get('declaration', '')
        buggy_body = item.get('buggy_solution', '')
        
        full_buggy_code = f"{imports}\n{declaration}{buggy_body}".strip()
        
        task = {
            "task_id": f"hep_{item['task_id']}",
            "problem": item['instruction'],
            "buggy_code": full_buggy_code,
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

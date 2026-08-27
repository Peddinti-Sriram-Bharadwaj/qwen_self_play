import json
import subprocess
import sys

def ensure_datasets_installed():
    try:
        import datasets
    except ImportError:
        print("HuggingFace 'datasets' library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def generate_mbpp_dataset():
    ensure_datasets_installed()
    from datasets import load_dataset
    
    print("Downloading MBPP dataset for function-level Python tasks...")
    # MBPP contains exactly the kind of short, self-contained Python functions with unit tests you requested
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized")
    
    tasks = []
    
    # We will grab 80 tasks to fulfill your 50/10/20 split requirement
    # We iterate over the dataset and format it into your exact requested JSON schema
    for i, item in enumerate(dataset["train"]):
        if i >= 80:
            break
            
        task = {
            "task_id": f"mbpp_{item['task_id']}",
            "problem": item['prompt'],
            "correct_code": item['code'],
            "tests": item['test_list']
        }
        tasks.append(task)
        
    # Split the dataset
    train_split = tasks[:50]
    val_split = tasks[50:60]
    test_split = tasks[60:80]
    
    output = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }
    
    with open("synthetic_tasks.json", "w") as f:
        json.dump(output, f, indent=4)
        
    print(f"Successfully generated 'synthetic_tasks.json'!")
    print(f"Breakdown: {len(train_split)} train, {len(val_split)} val, {len(test_split)} test.")
    
if __name__ == "__main__":
    generate_mbpp_dataset()

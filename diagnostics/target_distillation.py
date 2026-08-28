import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import json
import random

def generate_distillation_datasets(tokenizer):
    """
    Generates three static datasets of varying difficulty for the Control Target Test.
    1. Easy: Memorize a single simple repeated sentence.
    2. Medium: Memorize a 10-example subset of MBPP Task B.
    3. Hard: Memorize random token sequences (maximum capacity test).
    """
    datasets = {}
    
    # --- EASY: Single Sentence ---
    easy_text = "The quick brown fox jumps over the lazy dog. " * 10
    easy_tokens = tokenizer(easy_text, return_tensors="pt")["input_ids"][0][:100]
    datasets["easy"] = [easy_tokens] * 10  # 10 identical batches

    # --- MEDIUM: 10-shot MBPP Phase B ---
    with open("data/train_B.json", "r") as f:
        mbpp_b = json.load(f)[:10]
    medium_tokens = []
    for task in mbpp_b:
        text = f"Problem: {task['problem']}\nSolution:\n{task['correct_code']}"
        tokens = tokenizer(text, return_tensors="pt")["input_ids"][0][:100] # Truncate for speed
        medium_tokens.append(tokens)
    datasets["medium"] = medium_tokens

    # --- HARD: Random Token Sequences ---
    vocab_size = tokenizer.vocab_size
    hard_tokens = []
    for _ in range(10):
        # Generate 100 random tokens (simulating a completely unstructured target)
        tokens = torch.randint(0, vocab_size, (100,))
        hard_tokens.append(tokens)
    datasets["hard"] = hard_tokens

    return datasets

def run_target_distillation(model, dataset, device, steps=50):
    """
    Runs a fast supervised learning loop (Next-Token Prediction) on a static dataset.
    Returns the loss curve.
    """
    # Ensure model is in train mode (but only train the base weights or adapter weights depending on the model state)
    model.train()
    
    # Use a fresh optimizer for this diagnostic run (only optimizing trainable weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    loss_curve = []
    
    for step in tqdm(range(steps), desc="Control Target Distillation Steps"):
        total_loss = 0
        for seq in dataset:
            seq = seq.unsqueeze(0).to(device) # Batch size 1
            
            inputs = seq[:, :-1]
            targets = seq[:, 1:]
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            logits = outputs.logits
            
            # Reshape for CrossEntropyLoss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataset)
        loss_curve.append(avg_loss)
        
    return loss_curve

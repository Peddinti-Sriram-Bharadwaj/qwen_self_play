import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. Dataset Generation (Using Real HuggingFace Dataset)
# We use lmsys/toxic-chat which provides real-world user prompts annotated for toxicity.
def load_real_dataset(num_samples=500):
    print("Loading real 'toxic-chat' dataset from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    
    # Extract safe and unsafe prompts
    safe_prompts = [x['user_input'] for x in dataset if x['toxicity'] == 0][:num_samples]
    unsafe_prompts = [x['user_input'] for x in dataset if x['toxicity'] == 1][:num_samples]
    
    prompts = safe_prompts + unsafe_prompts
    labels = np.array([0]*len(safe_prompts) + [1]*len(unsafe_prompts))
    return prompts, labels

def extract_last_token_embeddings(model, tokenizer, prompts, batch_size=8):
    """Extracts the late-layer hidden state activation embeddings (last token)."""
    model.eval()
    all_embeddings = []
    
    print(f"Extracting embeddings for {len(prompts)} prompts...")
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        formatted_prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch_prompts
        ]
        
        inputs = tokenizer(formatted_prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Get the last hidden state: [batch, seq_len, dim]
        last_hidden = outputs.hidden_states[-1]
        
        # Get the embedding for the last token in each sequence
        seq_lengths = inputs.attention_mask.sum(dim=1) - 1
        
        batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        last_token_embeddings = last_hidden[batch_indices, seq_lengths, :]
        
        all_embeddings.append(last_token_embeddings.cpu().float().numpy())
        
    return np.vstack(all_embeddings)

def apply_spherical_perturbation(embeddings, sigma):
    """Applies controlled spherical perturbation (drift) to normalized embeddings."""
    if sigma == 0.0:
        return embeddings
        
    N, D = embeddings.shape
    drifted_embeddings = np.zeros_like(embeddings)
    
    for i in range(N):
        e_hat = embeddings[i]
        
        # 1. Sample random noise vector
        eta = np.random.normal(0, 1, D)
        
        # 2. Project onto orthogonal complement of e_hat
        dot_product = np.dot(eta, e_hat)
        eta_orth = eta - dot_product * e_hat
        
        # 3. Normalize and scale to sigma
        eta_orth_norm = np.linalg.norm(eta_orth)
        if eta_orth_norm > 0:
            eta_projected = sigma * (eta_orth / eta_orth_norm)
        else:
            eta_projected = np.zeros_like(eta_orth)
            
        # 4. Add perturbation and re-normalize to unit sphere
        e_drifted = e_hat + eta_projected
        e_drifted_norm = np.linalg.norm(e_drifted)
        drifted_embeddings[i] = e_drifted / e_drifted_norm
        
    return drifted_embeddings

def evaluate_classifier_at_scale(clf, X, y):
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    
    roc_auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, preds)
    
    # Calculate Mean Classifier Confidence
    confidences = np.abs(probs - 0.5) * 2
    mean_confidence = np.mean(confidences)
    
    # Calculate Silent Failure Rate
    misclassified_idx = np.where(preds != y)[0]
    if len(misclassified_idx) > 0:
        misclassified_probs = probs[misclassified_idx]
        high_conf_failures = np.sum((misclassified_probs > 0.8) | (misclassified_probs < 0.2))
        silent_failure_rate = high_conf_failures / len(misclassified_idx)
    else:
        silent_failure_rate = 0.0
        
    return roc_auc, mean_confidence, silent_failure_rate

def run_experiment(base_model_path, rl_checkpoints_dir=None):
    print("="*50)
    print(" SAHOO EXPERIMENT A: SPHERICAL EMBEDDING DRIFT")
    print("="*50)
    
    print(f"\nLoading model: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # 1. Load Real Dataset & Extract Embeddings
    PROMPTS, LABELS = load_real_dataset(num_samples=500)
    X = extract_last_token_embeddings(model, tokenizer, PROMPTS)
    
    # 2. Normalize to unit L2 norm
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms
    
    # 3. Split dataset
    X_train, X_eval, y_train, y_eval, _, eval_prompts = train_test_split(
        X_normalized, LABELS, PROMPTS, test_size=0.2, random_state=42
    )
    
    # 4. Train Baseline Safety Classifier
    print("\nTraining downstream safety classifier (Logistic Regression) on real dataset...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    base_auc, base_conf, _ = evaluate_classifier_at_scale(clf, X_eval, y_eval)
    print(f"Baseline (Clean) Eval - ROC-AUC: {base_auc:.3f}, Mean Confidence: {base_conf:.3f}")
    
    # 5. Sweep sigma and track metrics
    sigmas = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    aucs = []
    confs = []
    silent_fail_rates = []
    
    print("\nEvaluating Spherical Perturbations...")
    for sigma in sigmas:
        X_drifted = apply_spherical_perturbation(X_eval, sigma)
        
        auc, conf, sfr = evaluate_classifier_at_scale(clf, X_drifted, y_eval)
        aucs.append(auc)
        confs.append(conf)
        silent_fail_rates.append(sfr)
        
        print(f"  \u03c3 = {sigma:.3f} | ROC-AUC: {auc:.3f} | Mean Conf: {conf:.3f} | Silent Fail Rate: {sfr*100:.1f}%")
        
    # 6. Visualization
    plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Noise Scale (\u03c3)')
    ax1.set_ylabel('Classifier ROC-AUC', color=color)
    ax1.plot(sigmas, aucs, marker='o', color=color, linewidth=2, label='ROC-AUC')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0.4, 1.05])
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Mean Classifier Confidence', color=color)
    ax2.plot(sigmas, confs, marker='s', linestyle='--', color=color, linewidth=2, label='Mean Confidence')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0.0, 1.05])
    
    plt.title("Catastrophic Collapse of Safety Classifiers under Embedding Drift")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plot_path = "sahoo_drift_collapse_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nPlot saved to {plot_path}")
    
    # 7. Actual RL Checkpoints Evaluation
    if rl_checkpoints_dir:
        print(f"\n==================================================")
        print(f" Evaluating Actual RL Checkpoint Drift over Time")
        print(f" Directory: {rl_checkpoints_dir}")
        print(f"==================================================")
        
        del model
        torch.cuda.empty_cache()
        
        checkpoints = sorted(glob.glob(os.path.join(rl_checkpoints_dir, "phase*_step_*")))
        
        if not checkpoints:
            print("No checkpoints found in the directory.")
        else:
            for ckpt_path in checkpoints:
                step_name = os.path.basename(ckpt_path)
                print(f"\nLoading model: {step_name}...")
                
                drift_model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, 
                    torch_dtype=torch.bfloat16, 
                    device_map="auto"
                )
                
                print(f"Extracting RL drifted embeddings for {len(eval_prompts)} test prompts...")
                X_rl = extract_last_token_embeddings(drift_model, tokenizer, eval_prompts)
                
                # Normalize the RL embeddings
                X_rl_norms = np.linalg.norm(X_rl, axis=1, keepdims=True)
                X_rl_normalized = X_rl / X_rl_norms
                
                rl_auc, rl_conf, rl_sfr = evaluate_classifier_at_scale(clf, X_rl_normalized, y_eval)
                print(f"  [{step_name}] | ROC-AUC: {rl_auc:.3f} | Mean Conf: {rl_conf:.3f} | Silent Fail Rate: {rl_sfr*100:.1f}%")
                
                del drift_model
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--rl_checkpoints_dir", type=str, default="", help="Directory containing all RL checkpoints")
    args = parser.parse_args()
    
    run_experiment(args.base_model, args.rl_checkpoints_dir)

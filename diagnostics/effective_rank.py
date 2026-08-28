import torch
import numpy as np

def measure_effective_rank(model, dataset, device):
    """
    Measures the Effective Rank of the final hidden states of the model.
    A lower rank indicates representation collapse.
    """
    model.eval()
    
    hidden_states = []
    
    def get_final_hidden_hook():
        def hook(module, inputs, outputs):
            # outputs shape: (batch_size, seq_len, hidden_size)
            # We want to collect the hidden states for all tokens
            h = outputs.detach()
            # Flatten batch and seq_len
            h = h.view(-1, h.size(-1))
            hidden_states.append(h)
        return hook

    # Hook the final layer norm before the lm_head
    norm_layer = model.base_model.model.norm if hasattr(model, "base_model") else model.model.norm
    hook = norm_layer.register_forward_hook(get_final_hidden_hook())
    
    print("Running validation batches for Effective Rank calculation...")
    with torch.no_grad():
        for seq in dataset:
            seq = seq.unsqueeze(0).to(device)
            model(seq)
            
    hook.remove()
    
    # Concatenate all hidden states
    H = torch.cat(hidden_states, dim=0).float()
    
    # Subsample if too large to fit SVD in memory (e.g. max 10000 tokens)
    if H.size(0) > 10000:
        indices = torch.randperm(H.size(0))[:10000]
        H = H[indices]
        
    # Center the hidden states (covariance)
    H = H - H.mean(dim=0, keepdim=True)
    
    # Compute SVD
    # For large matrices, torch.linalg.svdvals is faster
    print("Computing SVD...")
    singular_values = torch.linalg.svdvals(H)
    
    # Calculate Effective Rank (Shannon Entropy of singular values)
    # 1. Normalize singular values to create a probability distribution
    p = (singular_values / singular_values.sum()).cpu().numpy()
    
    # 2. Filter out exact zeros to avoid log(0)
    p = p[p > 0]
    
    # 3. Calculate Shannon entropy
    entropy = -np.sum(p * np.log(p))
    
    # 4. Effective rank is exp(entropy)
    effective_rank = np.exp(entropy)
    
    return effective_rank

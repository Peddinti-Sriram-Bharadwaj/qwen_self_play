import torch
import numpy as np

def compute_linear_cka(X, Y):
    """
    Computes the Linear Centered Kernel Alignment (CKA) between two activation matrices.
    X, Y shapes: (num_examples, num_features)
    """
    # Center the matrices
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute dot products
    # Instead of instantiating N x N similarity matrices (which OOMs), 
    # we use the linear CKA formulation: tr(X X^T Y Y^T) = ||X^T Y||_F^2
    dot_prod = torch.norm(torch.matmul(X.t(), Y), p='fro') ** 2
    norm_X = torch.norm(torch.matmul(X.t(), X), p='fro')
    norm_Y = torch.norm(torch.matmul(Y.t(), Y), p='fro')
    
    cka = dot_prod / (norm_X * norm_Y)
    return cka.item()

def measure_representational_warp(base_model, tuned_model, dataset, device):
    """
    Measures layer-wise CKA similarity between the base model and a tuned model.
    """
    base_model.eval()
    tuned_model.eval()
    
    num_layers = len(base_model.model.layers)
    cka_scores = {}
    
    for layer_idx in range(num_layers):
        # We will collect activations for one layer at a time to save memory
        base_acts = []
        tuned_acts = []
        
        def get_hook(act_list):
            def hook(module, inputs, outputs):
                # outputs is a tuple, we want the first element (hidden states)
                h = outputs[0].detach()
                h = h.view(-1, h.size(-1))
                act_list.append(h)
            return hook

        hook_base = base_model.model.layers[layer_idx].register_forward_hook(get_hook(base_acts))
        hook_tuned = tuned_model.model.layers[layer_idx].register_forward_hook(get_hook(tuned_acts))
        
        print(f"Running Layer {layer_idx} CKA extraction...")
        with torch.no_grad():
            for seq in dataset:
                seq = seq.unsqueeze(0).to(device)
                base_model(seq)
                tuned_model(seq)
                
        hook_base.remove()
        hook_tuned.remove()
        
        # Concatenate and compute CKA
        X = torch.cat(base_acts, dim=0).float()
        Y = torch.cat(tuned_acts, dim=0).float()
        
        # Subsample if too large
        if X.size(0) > 10000:
            indices = torch.randperm(X.size(0))[:10000]
            X = X[indices]
            Y = Y[indices]
            
        cka = compute_linear_cka(X, Y)
        cka_scores[layer_idx] = cka
        
    return cka_scores

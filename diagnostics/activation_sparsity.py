import torch
import numpy as np

def measure_dormant_neurons(model, dataset, device, tau=0.1):
    """
    Measures the percentage of dormant neurons in the FFN layers of a Qwen model.
    A neuron is tau-dormant if its normalized activation score s_i <= tau.
    """
    model.eval()
    
    # Store the expected absolute activation per layer
    layer_activations = {}
    
    def get_activation_hook(layer_idx):
        def hook(module, inputs, outputs):
            # inputs is a tuple, we want the first element which is the input to down_proj
            # Shape: (batch_size, seq_len, intermediate_size)
            x = inputs[0].detach().abs()
            
            # Average over batch and seq_len
            mean_activation = x.mean(dim=(0, 1))
            
            if layer_idx not in layer_activations:
                layer_activations[layer_idx] = []
            layer_activations[layer_idx].append(mean_activation)
        return hook

    # Register hooks on the down_proj of every MLP layer
    # The input to down_proj is exactly the post-activation intermediate features!
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.mlp.down_proj.register_forward_hook(get_activation_hook(i))
        hooks.append(hook)
        
    print("Running validation batches for Dormant Neuron calculation...")
    with torch.no_grad():
        for seq in dataset:
            seq = seq.unsqueeze(0).to(device)
            model(seq)
            
    for hook in hooks:
        hook.remove()
        
    dormancy_rates = {}
    
    # Calculate s_i for each layer
    for layer_idx, acts in layer_activations.items():
        # acts is a list of tensors (one per batch)
        # Average over all batches to get E[|h_i|]
        expected_abs_h = torch.stack(acts).mean(dim=0)
        
        # Calculate mean activation across all neurons in the layer
        mean_layer_act = expected_abs_h.mean()
        
        if mean_layer_act.item() == 0:
            dormancy_rates[layer_idx] = 100.0
            continue
            
        # Calculate normalized score s_i
        s_i = expected_abs_h / mean_layer_act
        
        # Count dormant neurons (s_i <= tau)
        dormant_count = (s_i <= tau).sum().item()
        total_neurons = s_i.size(0)
        
        dormancy_rates[layer_idx] = (dormant_count / total_neurons) * 100.0
        
    return dormancy_rates

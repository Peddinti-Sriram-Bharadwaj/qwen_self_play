import torch
import torch.nn.functional as F

def compute_sequence_logprobs(model, query, response, pad_token_id):
    """
    Computes the sum of log probabilities for the response tokens, given the query context.
    """
    # Concatenate query and response to form the full input sequence
    input_ids = torch.cat([query, response], dim=0).unsqueeze(0) # Shape: [1, seq_len_q + seq_len_r]
    
    # Forward pass
    outputs = model(input_ids)
    
    if isinstance(outputs, tuple):
        logits = outputs[0] # TRL AutoModelForCausalLMWithValueHead returns a tuple (logits, _, values)
    else:
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]
    
    # The logit at index t predicts the token at index t+1.
    # Therefore, the logits predicting the response tokens start at len(query) - 1.
    # We want logits up to the second-to-last token to predict the last token.
    start_idx = len(query) - 1
    end_idx = len(query) + len(response) - 1
    
    response_logits = logits[0, start_idx:end_idx, :] # Shape: [seq_len_r, vocab_size]
    
    # Compute log probabilities
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # Gather the log probabilities of the actual chosen tokens in the response
    token_log_probs = log_probs.gather(dim=-1, index=response.unsqueeze(1)).squeeze(-1)
    
    # Mask out pad tokens
    mask = (response != pad_token_id).float()
    
    # Return the sum of log probabilities for the generated response
    return (token_log_probs * mask).sum()

def calculate_plasticity_metrics(latents_batch: list[torch.Tensor]) -> dict:
    """
    Computes Feature Variance and Dormant Neurons percentage from a batch of hidden states.
    latents_batch: List of tensors of shape (seq_len, hidden_dim).
    
    Citations:
    - Lyle et al., "Loss of Plasticity in Continual Deep Reinforcement Learning" (https://arxiv.org/abs/2303.07507)
    - Defines dormant neurons as those with an average absolute activation below a small threshold.
    """
    if not latents_batch:
        return {"plasticity/feature_variance": 0.0, "plasticity/dormant_neurons_pct": 0.0}
        
    # Filter out empty latents
    valid_latents = [lt for lt in latents_batch if lt.numel() > 0]
    if not valid_latents:
        return {"plasticity/feature_variance": 0.0, "plasticity/dormant_neurons_pct": 0.0}
        
    # Stack along the sequence/batch dimension -> (total_tokens, hidden_dim)
    all_latents = torch.cat(valid_latents, dim=0)
    
    # 1. Feature Variance: Variance of each neuron across the batch, then averaged over all neurons
    # If the variance collapses to 0, the network has lost capacity.
    neuron_variances = all_latents.var(dim=0)
    mean_feature_variance = neuron_variances.mean().item()
    
    # 2. Dormant Neurons (%): Neurons whose average absolute activation across the batch is very close to 0.
    # We use a threshold of 1e-3.
    mean_abs_activations = all_latents.abs().mean(dim=0)
    dormant_threshold = 1e-3
    dormant_count = (mean_abs_activations < dormant_threshold).sum().item()
    dormant_pct = (dormant_count / all_latents.shape[-1]) * 100.0
    
    return {
        "plasticity/feature_variance": mean_feature_variance,
        "plasticity/dormant_neurons_pct": dormant_pct
    }

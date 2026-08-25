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

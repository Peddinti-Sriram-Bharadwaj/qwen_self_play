import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch.optim as optim

class PPOTrainer:
    def __init__(self, agent, lr=1e-5):
        self.agent = agent
        # Assuming agent.model is a transformers CausalLM
        self.optimizer = optim.AdamW(self.agent.model.parameters(), lr=lr)
        
        # A simple linear layer to act as a value head for the skeleton.
        # We attach this to the extracted latents.
        hidden_size = self.agent.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.agent.device)
        
        # Using bfloat16 for better numerical stability during mock training
        dtype = torch.bfloat16 if self.agent.device != "cpu" else torch.float32
        self.value_head.to(dtype)
        
        self.value_optimizer = optim.AdamW(self.value_head.parameters(), lr=lr)

    def update(self, trajectories):
        """
        Mock PPO update step.
        In a real implementation, you would compute advantages, calculate the PPO clip loss 
        with a reference model, and update the value head. 
        Here we perform a basic backward pass to ensure the computational graph is connected
        from the reward through the value head and into the LLM latents.
        """
        self.agent.model.train()
        total_loss = 0.0
        
        for step in trajectories:
            reward = step['reward']
            
            # To simulate gradient flow into the LLM, we re-run the forward pass with gradients enabled.
            # (Note: agent.extract_latents uses no_grad for fast inference, so we re-compute here).
            
            # Format the prompt to include the action that was actually taken
            messages = [
                {"role": "system", "content": "You are playing a game of Tic-Tac-Toe. Respond with only a single digit corresponding to your chosen move (0-8)."},
                {"role": "user", "content": step['observation']}
            ]
            prompt = self.agent.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt += str(step['action'])
            
            inputs = self.agent.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.agent.device)
            
            outputs = self.agent.model(**inputs, output_hidden_states=True)
            # Get the latent representation of the last token
            current_latents = outputs.hidden_states[-1][0, -1, :]
            
            current_value = self.value_head(current_latents)
            
            # Value loss
            target_value = torch.tensor([reward], dtype=current_value.dtype, device=self.agent.device)
            value_loss = nn.functional.mse_loss(current_value, target_value)
            
            # Mock Policy loss (simulate passing gradients to LLM)
            # In real PPO: policy_loss = - advantage * log_prob
            # Here we use mean of latents * advantage as a mock objective
            advantage = reward - current_value.item()
            policy_loss = - (current_latents.mean() * advantage)
            
            loss = value_loss + policy_loss
            
            self.optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            loss.backward()
            
            # Clip gradients to prevent exploding weights
            torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.value_optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Performed PPO update on {len(trajectories)} steps.")
        return total_loss / max(1, len(trajectories))

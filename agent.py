import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from abc import ABC, abstractmethod
import re

class Agent(ABC):
    @abstractmethod
    def act(self, observation: str):
        """
        Should return:
        - action (int): The parsed action for the environment.
        - query_tensor (torch.Tensor): The tokenized observation.
        - response_tensor (torch.Tensor): The tokenized generation (just the action tokens).
        - cot_latents (torch.Tensor): The hidden states of the generated CoT tokens.
        """
        pass
    
    # We no longer need extract_latents in the Agent interface since TRL handles the value head natively

class LocalLLMAgent(Agent):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using bfloat16 if mps/cuda, else float32 for numerical stability
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        
        # We load the model with a value head natively via TRL
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def act(self, observation: str):
        self.model.eval()
        # Prompt model for Chain of Thought
        messages = [
            {"role": "system", "content": "You are playing a game of Tic-Tac-Toe. Think step by step. Write your thoughts, then output 'Action: X' where X is your final chosen move (a single digit 0-8)."},
            {"role": "user", "content": observation}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        query_tensor = inputs.input_ids[0]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=40, # Increased for CoT
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        
        # The generated output contains the prompt + response. TRL needs just the response.
        sequences = outputs.sequences
        response_tensor = sequences[0][len(query_tensor):]
        response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
        
        # Extract the hidden states of the generated CoT tokens
        cot_latents_list = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            for step_hidden_states in outputs.hidden_states:
                # step_hidden_states[-1] is the last layer. shape: (1, 1, hidden_dim)
                cot_latents_list.append(step_hidden_states[-1][0, 0, :])
                
        if cot_latents_list:
            cot_latents = torch.stack(cot_latents_list) # shape: (seq_len, hidden_dim)
        else:
            cot_latents = torch.empty(0)
        
        # Extract action digit from CoT response
        match = re.search(r'Action:\s*(\d)', response_text)
        if match:
            action = int(match.group(1))
        else:
            # Fallback
            fallback_match = re.search(r'\d', response_text)
            action = int(fallback_match.group()) if fallback_match else -1
            
        return action, query_tensor.cpu(), response_tensor.cpu(), cot_latents.cpu()

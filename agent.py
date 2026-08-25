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
        # Prompt model to just output the number
        messages = [
            {"role": "system", "content": "You are playing a game of Tic-Tac-Toe. Respond with only a single digit corresponding to your chosen move (0-8)."},
            {"role": "user", "content": observation}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        query_tensor = inputs.input_ids[0]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=3, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # The generated output contains the prompt + response. TRL needs just the response.
        response_tensor = outputs[0][len(query_tensor):]
        response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
        
        # Extract first digit found
        match = re.search(r'\d', response_text)
        if match:
            action = int(match.group())
        else:
            # Fallback random valid move or invalid move signal
            action = -1
            
        return action, query_tensor.cpu(), response_tensor.cpu()

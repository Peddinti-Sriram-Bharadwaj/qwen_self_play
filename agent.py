import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
import re

class Agent(ABC):
    @abstractmethod
    def act(self, observation: str) -> int:
        pass
    
    @abstractmethod
    def extract_latents(self, observation: str) -> torch.Tensor:
        pass

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
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        
        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def act(self, observation: str) -> int:
        self.model.eval()
        # Prompt model to just output the number
        messages = [
            {"role": "system", "content": "You are playing a game of Tic-Tac-Toe. Respond with only a single digit corresponding to your chosen move (0-8)."},
            {"role": "user", "content": observation}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=3, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract first digit found
        match = re.search(r'\d', response)
        if match:
            return int(match.group())
        else:
            # Fallback random valid move or invalid move signal
            return -1

    def extract_latents(self, observation: str) -> torch.Tensor:
        """
        Extracts the hidden states/embeddings from the model for the given observation.
        """
        self.model.eval()
        messages = [
            {"role": "system", "content": "You are playing a game of Tic-Tac-Toe. Respond with only a single digit corresponding to your chosen move (0-8)."},
            {"role": "user", "content": observation}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor is of shape (batch_size, sequence_length, hidden_size)
        # We extract the last layer's hidden state for the last token
        last_hidden_state = outputs.hidden_states[-1][0, -1, :] 
        return last_hidden_state.cpu() # Return to CPU for storage/training

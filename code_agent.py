import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import re

class DualLoraCodeAgent:
    """
    Wraps a causal language model with two distinct LoRA adapters ('generator' and 'fixer').
    Provides utilities for fast adapter switching and batched generation.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device=None, lora_r=16):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load base model in bfloat16 to fit nicely on standard GPUs
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        
        # 1. Initialize PEFT configuration for the Generator
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # targeting attention weights is standard
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        print("Attaching 'generator' LoRA adapter...")
        self.model = get_peft_model(base_model, lora_config, adapter_name="generator")
        
        # 2. Add the second adapter for the Fixer
        print("Attaching 'fixer' LoRA adapter...")
        self.model.add_adapter("fixer", lora_config)
        
        # Default to generator active
        self.model.set_adapter("generator")
        print("Model initialized successfully with dual adapters.")
        
    def set_active_role(self, role: str):
        """Switches the active LoRA adapter (zero-cost pointer swap in PEFT)."""
        if role not in ["generator", "fixer"]:
            raise ValueError("Role must be 'generator' or 'fixer'")
        self.model.set_adapter(role)
        
    def batched_generate(self, prompts: list, adapter_name: str = "generator", max_tokens=256, temperature=0.7) -> list:
        """
        Generates responses for a batch of prompts using the specified adapter.
        """
        self.set_active_role(adapter_name)
        self.model.eval()
        
        inputs = self.tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        batch_size = len(prompts)
        responses = []
        for i in range(batch_size):
            # Extract just the generated tokens (ignoring the prompt)
            response_tensor = outputs[i][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            responses.append(response_text)
            
        return responses

def extract_python_code(text: str) -> str:
    """
    Extracts the first python code block from a generation.
    Falls back to the raw text if no markdown block is found.
    """
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

if __name__ == "__main__":
    # Simple compilation check (will only load model if run directly)
    pass

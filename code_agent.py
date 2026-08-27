import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class DualLoraCodeAgent:
    """
    Wraps a causal language model with two distinct LoRA adapters ('generator' and 'fixer').
    Provides utilities for fast adapter switching and batched generation.
    Code parsing utilities live in parsing.py (decoupled from the model).
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device=None, lora_r=16):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load base model in appropriate dtype
        dtype = torch.bfloat16 if self.device != "mps" else torch.float16
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(self.device)

        # 1. Initialize PEFT configuration for the Generator
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        print("Attaching 'generator' LoRA adapter...")
        self.model = get_peft_model(base_model, lora_config, adapter_name="generator")

        # 2. Add the second adapter for the Fixer
        print("Attaching 'fixer' LoRA adapter...")
        self.model.add_adapter("fixer", lora_config)

        # 3. Add the reference adapter for KL anchoring to a recent moving model
        print("Attaching 'fixer_ref' LoRA adapter...")
        self.model.add_adapter("fixer_ref", lora_config)
        
        print("Attaching 'generator_ref' LoRA adapter...")
        self.model.add_adapter("generator_ref", lora_config)

        # Default to generator active
        self.model.set_adapter("generator")
        print("Model initialized successfully with four adapters (generator, fixer, generator_ref, fixer_ref).")

    def set_active_role(self, role: str):
        """Switches the active LoRA adapter (zero-cost pointer swap in PEFT)."""
        if role not in ["generator", "fixer", "fixer_ref", "generator_ref"]:
            raise ValueError("Role must be 'generator', 'fixer', 'fixer_ref', or 'generator_ref'")
        self.model.set_adapter(role)

    def sync_fixer_ref(self):
        """
        Copies the current weights of the 'fixer' adapter into 'fixer_ref'.
        Used to update the KL divergence anchor to a recent policy.
        """
        print("Syncing 'fixer' weights into 'fixer_ref'...")
        self._sync_adapter_weights("fixer", "fixer_ref")
        
    def sync_generator_ref(self):
        """
        Copies the current weights of the 'generator' adapter into 'generator_ref'.
        """
        print("Syncing 'generator' weights into 'generator_ref'...")
        self._sync_adapter_weights("generator", "generator_ref")

    def _sync_adapter_weights(self, source: str, target: str):
        # Get state dicts for source adapter
        source_sd = {k: v for k, v in self.model.state_dict().items() if "lora_" in k and source in k and target not in k}
        
        # Construct the corresponding keys for target and load them
        ref_sd = {}
        for k, v in source_sd.items():
            ref_k = k.replace(source, target)
            ref_sd[ref_k] = v.clone().detach() # Clone to avoid memory reference sharing
            
        # Load the updated state dict into the model (strict=False to only update target keys)
        self.model.load_state_dict(ref_sd, strict=False)

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
            response_tensor = outputs[i][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
            responses.append(response_text)

        return responses


if __name__ == "__main__":
    # Simple compilation check (will only load model if run directly)
    pass

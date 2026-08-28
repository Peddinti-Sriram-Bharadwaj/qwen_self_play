import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class SelfPlayCodeAgent:
    """
    Wraps a causal language model for True Self-Play.
    Provides utilities for batched generation and abstract KL anchor computation.
    
    Supports two modes:
    - use_lora=True: Uses a single shared PEFT adapter for both generation and fixing.
    - use_lora=False: Makes the full base model trainable, and loads a separate frozen 
      reference model into memory for KL anchoring.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device=None, lora_r=16, use_lora=True):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.use_lora = use_lora
        print(f"Loading {model_name} on {self.device}... (use_lora={use_lora})")
        
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

        if self.use_lora:
            # 1. Initialize PEFT configuration for True Self-Play (Shared Adapter)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            print("Attaching shared 'self_play' LoRA adapter...")
            self.model = get_peft_model(base_model, lora_config, adapter_name="self_play")
            self.model.set_adapter("self_play")
            self.ref_model = None
        else:
            # 2. Full Adaptation (No PEFT)
            print("Preparing Full Adaptation. Base model weights are trainable.")
            self.model = base_model
            # Ensure model requires grad
            for param in self.model.parameters():
                param.requires_grad = True
                
            # Must load a separate frozen model for the KL reference anchor
            print("Loading separate frozen reference model for KL anchoring...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
            ).to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        print("Model initialized successfully.")

    def get_reference_logits(self, input_ids, attention_mask=None):
        """
        Abstracts the KL reference computation.
        If using LoRA, disables the adapter temporarily.
        If using Full Adaptation, routes the forward pass to the frozen ref_model.
        """
        with torch.no_grad():
            if self.use_lora:
                with self.model.disable_adapter():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self.ref_model(input_ids, attention_mask=attention_mask)
        return outputs.logits

    def batched_generate(self, prompts: list, max_tokens=256, temperature=0.7) -> list:
        """
        Generates responses for a batch of prompts using the shared model weights.
        """
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
    pass

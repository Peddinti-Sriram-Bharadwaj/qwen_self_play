import os
import random
import torch
from agent import LocalLLMAgent

class OpponentManager:
    """
    Manages the opponent pool for Multi-Agent Self-Play Regimes.
    Regime A: Fixed Opponent (Untrained baseline)
    Regime B: Evolving Opponent (Current policy)
    Regime C: Evolving Opponent + High Replay
    Regime D: Historical Opponent Pool (FSP / League)
    """
    def __init__(self, regime: str, learning_agent: LocalLLMAgent, checkpoint_dir: str = "checkpoints"):
        self.regime = regime
        self.learning_agent = learning_agent
        self.checkpoint_dir = checkpoint_dir
        self.historical_pool = []
        self.batch_opponent = None
        self.save_schedule = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
        
        print(f"Initializing Opponent Shell (Frozen Base Policy)...")
        # Extract the huggingface model path
        model_name = getattr(self.learning_agent.model.config, "_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        self.shell_opponent = LocalLLMAgent(model_name=model_name, device=self.learning_agent.device)
        self.shell_opponent.model.eval()
        self.shell_opponent.model.requires_grad_(False)
        
        if self.regime == "D":
                # Save the frozen base immediately to the historical pool (checkpoint_0)
                self.save_checkpoint(0, force=True)
            
    def save_checkpoint(self, iteration: int, force: bool = False):
        if self.regime == "D":
            if force or iteration in self.save_schedule:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pt")
                torch.save(self.learning_agent.model.state_dict(), path)
                self.historical_pool.append(path)
                print(f"[OpponentManager] Saved historical checkpoint to {path}. Pool size: {len(self.historical_pool)}")
            
    def prepare_batch_opponent(self):
        """
        Called once per data-collection phase (e.g. at the start of self_play collection)
        to load the weights for the opponent into VRAM for the entire batch.
        """
        if self.regime in ["B", "C"]:
            self.batch_opponent = self.learning_agent
        elif self.regime == "A":
            # shell_opponent retains its original random/untrained weights
            self.batch_opponent = self.shell_opponent
        elif self.regime == "D":
            # FSP: Sample uniformly from historical checkpoints
            if not self.historical_pool:
                self.batch_opponent = self.learning_agent
            else:
                ckpt_path = random.choice(self.historical_pool)
                ckpt = torch.load(ckpt_path, weights_only=True)
                # TRL's AutoModelForCausalLMWithValueHead strips 'pretrained_model.' prefix on save
                # but expects it on load_state_dict. We remap it here.
                mapped_ckpt = {}
                for k, v in ckpt.items():
                    if not k.startswith("v_head.") and not k.startswith("pretrained_model."):
                        mapped_ckpt[f"pretrained_model.{k}"] = v
                    else:
                        mapped_ckpt[k] = v
                self.shell_opponent.model.load_state_dict(mapped_ckpt)
                self.batch_opponent = self.shell_opponent
                
    def get_opponent(self):
        return self.batch_opponent

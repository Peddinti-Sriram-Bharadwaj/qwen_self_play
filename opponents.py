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
        
        if self.regime in ["A", "D"]:
            print(f"Initializing Opponent Shell (Regime {self.regime})...")
            # Extract the huggingface model path
            model_name = getattr(self.learning_agent.model.config, "_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
            self.shell_opponent = LocalLLMAgent(model_name=model_name, device=self.learning_agent.device)
            self.shell_opponent.model.eval()
            self.shell_opponent.model.requires_grad_(False)
            
    def save_checkpoint(self, iteration: int):
        if self.regime == "D":
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration}.pt")
            torch.save(self.learning_agent.model.state_dict(), path)
            self.historical_pool.append(path)
            
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
                # If no checkpoints exist yet, play against current self
                self.batch_opponent = self.learning_agent
            else:
                ckpt_path = random.choice(self.historical_pool)
                print(f"[OpponentManager] Loaded historical opponent: {ckpt_path}")
                self.shell_opponent.model.load_state_dict(torch.load(ckpt_path))
                self.batch_opponent = self.shell_opponent
                
    def get_opponent(self):
        return self.batch_opponent

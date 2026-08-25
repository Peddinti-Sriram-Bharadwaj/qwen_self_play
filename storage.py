import os
import torch
from pathlib import Path

class TrajectoryStorage:
    def __init__(self, base_dir="latents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.episode_counter = 0

    def start_new_episode(self):
        self.episode_counter += 1
        self.step_counter = 0
        self.current_episode_dir = self.base_dir / f"episode_{self.episode_counter:04d}"
        self.current_episode_dir.mkdir(exist_ok=True)

    def save_step_latents(self, player: int, cot_latents: torch.Tensor):
        """
        Saves the CoT latents for a specific step to disk.
        cot_latents: shape (seq_len, hidden_dim)
        """
        self.step_counter += 1
        # Save as a standard PyTorch .pt file.
        # For a massive scale, HDF5 or safetensors is better, but .pt is fine for the skeleton.
        file_path = self.current_episode_dir / f"step_{self.step_counter:04d}_player_{player}.pt"
        torch.save(cot_latents.clone().detach().cpu(), file_path)

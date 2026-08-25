import os
import torch
from pathlib import Path

class TrajectoryStorage:
    def __init__(self, base_dir="latents"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.episode_counter = 0

    def start_new_episode(self) -> str:
        self.episode_counter += 1
        episode_id = f"episode_{self.episode_counter:04d}"
        current_episode_dir = self.base_dir / episode_id
        current_episode_dir.mkdir(exist_ok=True)
        return episode_id

    def save_step_latents(self, episode_id: str, step: int, player: int, cot_latents: torch.Tensor):
        """
        Saves the CoT latents for a specific step to disk.
        cot_latents: shape (seq_len, hidden_dim)
        """
        episode_dir = self.base_dir / episode_id
        # Save as a standard PyTorch .pt file.
        file_path = episode_dir / f"step_{step:04d}_player_{player}.pt"
        torch.save(cot_latents.clone().detach().cpu(), file_path)

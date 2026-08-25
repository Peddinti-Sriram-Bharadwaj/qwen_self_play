import os
import csv
import json
import subprocess
import wandb
from datetime import datetime

class MetricsLogger:
    def __init__(self, config: dict, base_dir: str = "runs"):
        self.config = config
        self.env_name = config.get("environment", "unknown_env")
        self.regime = config.get("regime", "unknown_regime")
        self.seed = config.get("seed", 0)
        
        # Create directory structure: runs/<env>/regime_<X>/seed_<Y>/
        self.run_dir = os.path.join(base_dir, self.env_name, f"regime_{self.regime}", f"seed_{self.seed}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Save Git Commit
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
            with open(os.path.join(self.run_dir, "git_commit.txt"), "w") as f:
                f.write(commit_hash)
        except Exception as e:
            print(f"Warning: Could not save git commit: {e}")
            
        # Save Config
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=4)
            
        # Initialize CSVs
        self.metrics_csv = os.path.join(self.run_dir, "metrics.csv")
        self.eval_csv = os.path.join(self.run_dir, "evaluation.csv")
        self._metrics_headers_written = False
        self._eval_headers_written = False
        
        # Initialize WandB
        if config.get("use_wandb", True):
            wandb.init(
                project=config.get("project", "continual-self-play"),
                name=f"regime_{self.regime}_seed{self.seed}",
                group=f"{self.env_name}-main",
                config=config,
                tags=[self.regime, self.env_name, config.get("backend", "unknown"), config.get("algorithm", "DAPO")]
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
            
    def log_metrics(self, iteration: int, metrics: dict):
        """Logs scalar metrics during training"""
        flat_metrics = {"iteration": iteration}
        flat_metrics.update(metrics)
        
        # Wandb
        if self.use_wandb:
            wandb.log(flat_metrics, step=iteration)
            
        # CSV
        file_exists = os.path.isfile(self.metrics_csv)
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_metrics.keys())
            if not self._metrics_headers_written and not file_exists:
                writer.writeheader()
                self._metrics_headers_written = True
            try:
                writer.writerow(flat_metrics)
            except ValueError:
                pass
                
    def log_evaluation(self, iteration: int, opponent_name: str, results: dict):
        """Logs cross-play evaluation results"""
        row = {"iteration": iteration, "opponent": opponent_name}
        row.update(results)
        
        if self.use_wandb:
            wandb_metrics = {f"eval_{opponent_name}/{k}": v for k, v in results.items()}
            wandb.log(wandb_metrics, step=iteration)
            
        file_exists = os.path.isfile(self.eval_csv)
        with open(self.eval_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._eval_headers_written and not file_exists:
                writer.writeheader()
                self._eval_headers_written = True
                
            try:
                writer.writerow(row)
            except ValueError:
                pass
                
    def finish(self):
        if self.use_wandb:
            wandb.finish()

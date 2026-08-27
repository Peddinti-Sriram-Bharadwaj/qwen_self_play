import os
import sys
import argparse

# Parse the GPU argument BEFORE importing PyTorch to guarantee strict VRAM isolation
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", type=str, default="0", help="Pin to a specific GPU (sets CUDA_VISIBLE_DEVICES)")
_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _args.gpu

import json
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from code_agent import DualLoraCodeAgent, extract_python_code
from sandbox import evaluate_code
import random
import numpy as np
import wandb

class GRPOSelfPlayLoop:
    """
    Implements the Anchored Self-Play reinforcement learning loop using GRPO.
    Handles the asynchronous sampling and training of the Generator and Fixer.
    """
    def __init__(self, agent: DualLoraCodeAgent, dataset_path="synthetic_tasks.json", lr=1e-5, beta=0.04):
        self.agent = agent
        self.optimizer = AdamW(self.agent.model.parameters(), lr=lr)
        self.beta = beta
        
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
            
        self.train_tasks = self.dataset["train"]
        
        # Generator prompt template
        self.gen_prompt = """You are a code bug generator. Here is a correctly working function and its unit tests.
Your task is to introduce a subtle bug into this function such that it fails the unit tests, but remains syntactically valid Python.
Output ONLY the buggy Python code block.

Problem: {problem}
Tests: {tests}

Correct Code:
```python
{code}
```
"""

        # Fixer prompt template
        self.fix_prompt = """You are an expert Python debugger. Here is a problem description, a suite of unit tests, and a buggy implementation.
Your task is to fix the bug so that all tests pass.
Output ONLY the corrected Python code block.

Problem: {problem}
Tests: {tests}

Buggy Code:
```python
{buggy_code}
```
"""

    def compute_grpo_loss(self, prompts, responses, advantages, adapter_name, beta=None, epsilon=0.2):
        """
        Computes the Group Relative Policy Optimization loss using standard PyTorch.
        """
        if beta is None:
            beta = self.beta
            
        self.agent.set_active_role(adapter_name)
        self.agent.model.train()
        
        total_loss = 0
        
        for prompt, response, adv in zip(prompts, responses, advantages):
            # 1. Tokenize sequence
            prompt_ids = self.agent.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.agent.device)
            response_ids = self.agent.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.agent.device)
            
            seq_ids = torch.cat([prompt_ids, response_ids], dim=1)
            prompt_len = prompt_ids.shape[1]
            
            # 2. Forward pass: Active Policy
            outputs = self.agent.model(seq_ids)
            logits = outputs.logits[:, :-1, :] # Shift logits for next-token prediction
            target_ids = seq_ids[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            
            # Extract logprobs only for the generated response
            pi_theta_logprobs = token_logprobs[:, prompt_len-1:]
            
            # 3. Forward pass: Reference Policy (Base Model)
            with torch.no_grad():
                with self.agent.model.disable_adapter():
                    ref_outputs = self.agent.model(seq_ids)
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_logprobs = ref_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
                    pi_ref_logprobs = ref_token_logprobs[:, prompt_len-1:]
                    
            # 4. Compute GRPO PPO-Clip Loss
            ratio = torch.exp(pi_theta_logprobs - pi_ref_logprobs)
            
            clip_adv = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            unclip_adv = ratio * adv
            
            # Negative because optimizer minimizes, but we want to maximize reward
            policy_loss = -torch.min(clip_adv, unclip_adv).mean()
            
            # 5. KL Divergence Penalty (approximated)
            kl_div = torch.exp(pi_ref_logprobs - pi_theta_logprobs) - (pi_ref_logprobs - pi_theta_logprobs) - 1
            kl_loss = kl_div.mean()
            
            loss = policy_loss + beta * kl_loss
            total_loss += loss
            
        # Average loss over batch
        avg_loss = total_loss / len(prompts)
        
        # Accumulate gradients
        avg_loss.backward()
        
        return avg_loss.item()

    def run_step(self, G=4, K=4):
        """Executes a single step of the Self-Play loop."""
        # 1. Sample task
        task = random.choice(self.train_tasks)
        tests_str = "\n".join(task["tests"])
        
        print(f"\n--- STEP: Task {task['task_id']} ---")
        
        # 2. Generator phase
        gen_prompts = [self.gen_prompt.format(
            problem=task["problem"], 
            tests=tests_str, 
            code=task["correct_code"]
        )] * G
        
        print("Sampling Generator...")
        raw_bugs = self.agent.batched_generate(gen_prompts, adapter_name="generator", max_tokens=150)
        parsed_bugs = [extract_python_code(b) for b in raw_bugs]
        
        # Filter for valid bugs
        valid_bugs = []
        for bug in parsed_bugs:
            res = evaluate_code(bug, task["tests"])
            if res["executable"] and res["has_assertion_failure"]:
                valid_bugs.append(bug)
                
        if not valid_bugs:
            print("Generator failed to produce any valid bugs this step (e.g. syntax errors or tests still pass). Skipping.")
            return
            
        print(f"Generated {len(valid_bugs)} valid bugs out of {G}.")
            
        # 3. Fixer Phase
        target_bug = valid_bugs[0] # Pick the first valid bug for the fixer to repair
        fix_prompts = [self.fix_prompt.format(
            problem=task["problem"],
            tests=tests_str,
            buggy_code=target_bug
        )] * K
        
        print("Sampling Fixer...")
        raw_fixes = self.agent.batched_generate(fix_prompts, adapter_name="fixer", max_tokens=150)
        parsed_fixes = [extract_python_code(f) for f in raw_fixes]
        
        # Evaluate fixes
        fix_rewards = []
        for fix in parsed_fixes:
            res = evaluate_code(fix, task["tests"])
            fix_rewards.append(1.0 if res["all_passed"] else 0.0)
            
        fix_rewards = np.array(fix_rewards)
        empirical_fix_rate = np.mean(fix_rewards)
        print(f"Fixer Success Rate: {empirical_fix_rate*100}%")
            
        # 4. Compute GRPO Advantages
        if np.std(fix_rewards) > 1e-8:
            fix_advantages = (fix_rewards - np.mean(fix_rewards)) / (np.std(fix_rewards) + 1e-8)
        else:
            fix_advantages = fix_rewards - np.mean(fix_rewards)
            
        # 5. Fixer Loss Update
        self.optimizer.zero_grad()
        print("Computing Fixer GRPO Loss...")
        fixer_loss = self.compute_grpo_loss(
            prompts=fix_prompts,
            responses=raw_fixes, # Must use the raw text output to align tokens exactly
            advantages=torch.tensor(fix_advantages, dtype=torch.float32).to(self.agent.device),
            adapter_name="fixer"
        )
        
        # 6. Generator Reward & Loss Update
        # The generator gets a positive reward if the fixer succeeds between 25% and 75% of the time.
        # If it's too easy (1.0) or too hard (0.0), it gets penalized.
        if 0.25 <= empirical_fix_rate <= 0.75:
            gen_reward = 1.0
        else:
            gen_reward = -0.5
            
        # Apply this reward as the advantage for the generator's samples this step
        gen_advantages = torch.full((G,), gen_reward, dtype=torch.float32).to(self.agent.device)
        
        print("Computing Generator GRPO Loss...")
        gen_loss = self.compute_grpo_loss(
            prompts=gen_prompts,
            responses=raw_bugs,
            advantages=gen_advantages,
            adapter_name="generator"
        )
        
        # 7. Step the Optimizer
        self.optimizer.step()
        print(f"Step Complete. Fixer Loss: {fixer_loss:.4f} | Generator Loss: {gen_loss:.4f}")
        
        return {
            "fixer_loss": fixer_loss,
            "generator_loss": gen_loss,
            "fixer_success_rate": empirical_fix_rate,
            "generator_reward": gen_reward
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Anchored Self-Play")
    parser.add_argument("--beta", type=float, default=0.04, help="KL Divergence Penalty (set to 0.0 for unanchored collapse)")
    parser.add_argument("--run-name", type=str, default="qwen1.5b_anchored_run", help="WandB run name")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to pin to (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--K", type=int, default=16, help="Number of fixer samples per bug (higher = smoother charts)")
    parser.add_argument("--G", type=int, default=4, help="Number of generator samples per task")
    args = parser.parse_args()

    print("Initializing Dual LoRA Agent (Qwen2.5-Coder-1.5B)...")
    agent = DualLoraCodeAgent(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    
    print(f"Initializing GRPO Self-Play Loop (Beta = {args.beta})...")
    loop = GRPOSelfPlayLoop(agent=agent, dataset_path="data/synthetic_tasks.json", lr=1e-5, beta=args.beta)
    
    wandb.init(project="anchored_code_self_play", name=args.run_name, config={"beta": args.beta})
    
    print("Starting Anchored Self-Play Training...")
    for step in range(1, args.steps + 1):
        metrics = loop.run_step(G=args.G, K=args.K)
        if metrics:
            wandb.log(metrics, step=step)
        
        # Save checkpoints periodically
        if step % 50 == 0:
            print(f"Saving checkpoints at step {step}...")
            agent.set_active_role("generator")
            agent.model.save_pretrained(f"checkpoints/step_{step}_generator")
            agent.set_active_role("fixer")
            agent.model.save_pretrained(f"checkpoints/step_{step}_fixer")

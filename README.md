# Anchored Self-Play for Continual Code Repair

This branch implements a continual learning experiment that investigates whether a KL-anchored reinforcement learning loop — structured as adversarial self-play between a bug generator and a bug fixer — can resist catastrophic forgetting when fine-tuned sequentially across two distinct task distributions.

## Research Question

Standard RL fine-tuning on sequential task distributions causes catastrophic forgetting: performance on earlier distributions degrades as the model adapts to later ones [McCloskey & Cohen, 1989; Kirkpatrick et al., 2017]. A common mitigation in RLHF is to constrain the policy with a KL penalty relative to a fixed reference model [Ziegler et al., 2019; Ouyang et al., 2022], which we refer to as *anchoring*.

We ask: does anchoring the self-play policy to the pre-trained base model preserve performance on Distribution A while training on Distribution B, and vice versa?

## Method

### Self-Play Loop

The model maintains two separate LoRA adapters [Hu et al., 2022] attached to a single frozen base model (`Qwen/Qwen2.5-Coder-1.5B-Instruct`):

- **Generator** adapter: given a correct solution, introduces a subtle bug that causes unit tests to fail while remaining syntactically valid Python.
- **Fixer** adapter: given a buggy solution and its unit tests, produces a corrected implementation.

At each training step, the generator produces G candidate bugs. Valid bugs — those that are executable but fail the test suite — are passed to the fixer, which produces K candidate repairs. Both adapters are updated via Group Relative Policy Optimization (GRPO) [Shao et al., 2024].

### Reward Design

The fixer receives a shaped reward based on the execution outcome in an isolated subprocess sandbox:

| Outcome | Reward |
|--------|--------|
| All tests pass | +1.0 |
| AssertionError (logic wrong, code runs) | 0.0 |
| Timeout (infinite loop) | -0.5 |
| Syntax or runtime crash | -1.0 |

The generator receives a "goldilocks" reward: +1.0 if the fixer success rate falls in [25%, 75%], and -0.5 otherwise. This incentivizes bugs of appropriate difficulty — neither trivially easy nor impossible — producing an emergent curriculum [Sukhbaatar et al., 2017].

### KL Anchoring

Following standard RLHF practice [Ziegler et al., 2019], the policy loss includes a KL divergence penalty relative to the base model (no adapters active):

```
L = L_CLIP(pi_theta, pi_ref, A) + beta * KL(pi_ref || pi_theta)
```

The KL approximation follows Schulman's k3 estimator. Setting `beta=0.0` removes the anchor and recovers standard GRPO, enabling an ablation that isolates the contribution of anchoring.

### Experimental Protocol

- **Phase 0**: Evaluate the base model (no training) on both distributions to establish pass@1 baselines.
- **Phase 1**: Train on Distribution A (string/list/text-manipulation tasks from MBPP) for N steps. Evaluate on both distributions at fixed checkpoints.
- **Phase 2**: Train on Distribution B (numeric/mathematical tasks from MBPP) for N steps. Evaluate on both distributions at fixed checkpoints.

Evaluation uses `pass@1` on held-out bug-fix tasks from `bigcode/humanevalpack` [Muennighoff et al., 2023], executed in an isolated subprocess environment.

## Repository Structure

```
code_agent.py              # DualLoraCodeAgent: base model + two LoRA adapters
code_self_play.py          # GRPOSelfPlayLoop: one training step of the self-play loop
evaluate.py                # evaluate_checkpoint: pass@1 evaluation over a dataset
main_experiment.py         # Experiment orchestrator (Phase 0 → 1 → 2)
sandbox.py                 # Subprocess code execution + SandboxResult dataclass
prompts.py                 # Prompt templates (single source of truth for train and eval)
parsing.py                 # extract_python_code: post-processing of LLM output
rewards.py                 # fixer_reward, generator_reward: reward shaping functions
generate_strict_datasets.py # Downloads MBPP (train) and HumanEvalPack (eval) datasets
test_wiring.py             # Integration tests using a MockAgent (no GPU required)
```

## Installation

```bash
git clone https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play.git
cd qwen_self_play
git checkout anchored-self-play
conda activate rl_sim
python generate_strict_datasets.py   # downloads and splits data into data/
```

## Running the Experiment

```bash
# Full run (Phase 0 + 1 + 2)
nohup python -u main_experiment.py \
  --gpu 0 --phase_steps 200 --eval_freq 100 --K 4 --G 4 \
  > run_experiment.log 2>&1 &

# Resume after a crash, skipping Phase 0 re-evaluation
nohup python -u main_experiment.py \
  --gpu 0 --phase_steps 200 --eval_freq 100 --K 4 --G 4 \
  --base_eval_A 0.4375 --base_eval_B 0.39 \
  > run_experiment.log 2>&1 &
```

Metrics are logged to Weights & Biases under the project `anchored_code_self_play`. Baseline pass@1 values are logged at every training step as flat reference lines.

## Running the Wiring Test

The integration test exercises the full evaluation pipeline (prompts, parsing, sandbox, rewards, evaluate loop) using a `MockAgent` that returns pre-defined strings. No GPU is required.

```bash
python test_wiring.py
```

Expected output: 22/22 tests passing.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--phase_steps` | 200 | Training steps per distribution |
| `--eval_freq` | 100 | Evaluate every N steps |
| `--K` | 4 | Fixer samples per bug (higher = lower variance advantages) |
| `--G` | 4 | Generator samples per task |
| `--beta` | 0.04 | KL penalty coefficient (0.0 = unanchored ablation) |

## References

- Hu, E. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- McCloskey, M. & Cohen, N.L. (1989). *Catastrophic interference in connectionist networks.* Psychology of Learning and Motivation, 24, 109–165.
- Muennighoff, N. et al. (2023). *OctoPack: Instruction Tuning Code Large Language Models.* [arXiv:2308.07124](https://arxiv.org/abs/2308.07124)
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback.* NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Sukhbaatar, S. et al. (2017). *Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play.* ICLR 2018. [arXiv:1703.05407](https://arxiv.org/abs/1703.05407)
- Ziegler, D.M. et al. (2019). *Fine-Tuning Language Models from Human Preferences.* [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)

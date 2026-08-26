# Continual RL & Plasticity Collapse in LLM Self-Play

This repository investigates representation collapse (loss of plasticity) in Large Language Models (LLMs) when trained via continuous multi-agent self-play. 

It aims to test the hypothesis that shifting task distributions in zero-sum self-play can reduce a network's ability to adapt, characterized by a decrease in feature variance and an increase in dormant neurons. We implement and evaluate several mechanisms, such as experience replay and league training, to mitigate this effect.

### Citations & Prior Work
This experimental harness and the implemented plasticity tracking metrics reference the following work:
1. **[Loss of Plasticity in Continual Deep Reinforcement Learning (Lyle et al.)](https://arxiv.org/abs/2303.07507)** - Evaluates agents losing the ability to learn after cycling through changing tasks due to representation collapse.
2. **[SPIRAL](https://arxiv.org/abs/2506...)** - Related work on self-play and representation collapse.

## The Core Experiment

We evaluate the `Qwen/Qwen2.5-0.5B-Instruct` model on **Kuhn Poker** (via OpenSpiel) under four training regimes. Each regime trains for 2,000 iterations to measure performance and representation stability.

### The 4 Regimes
1. **Regime A (Fixed Opponent / Base Policy):** The learning agent plays against a frozen, untrained version of itself. This establishes a single-agent baseline for maintaining plasticity on a stationary task.
2. **Regime B (Evolving Opponent / Current Policy):** Standard self-play. The agent plays against the current version of itself. This evaluates the effect of shifting tasks in self-play on plasticity.
3. **Regime C (Evolving Opponent + High Replay):** The agent plays against itself but samples off-policy data from a buffer of 50,000 past experiences to stabilize the data distribution.
4. **Regime D (Historical Opponent Pool / League Training):** Fictitious Self-Play (FSP). The agent plays against randomly selected past versions of itself. This broadens the task distribution without relying on replay buffers.

### Plasticity Tracking
We measure the LLM's final-layer hidden states (`cot_latents`) across updates to track two metrics:
- **Feature Variance:** The variance of the activations. A decrease indicates a potential loss of capacity.
- **Dormant Neurons:** The percentage of neurons with a mean absolute activation `< 1e-3`.

## Installation

We provide an automated setup script (`install.sh`) that works on both macOS (MPS) and Linux Servers (CUDA 12.1).

```bash
git clone https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play.git
cd qwen_self_play
git checkout experimental-harness

chmod +x install.sh
./install.sh
```

## Running the Main Experiments

You can execute the main experiment loops (2,000 iterations for 3 seeds each) across all regimes. 

> [!TIP]  
> If you are on a shared server, use `htop -u $USER` to monitor your CPU/RAM usage. If you need to stop your runs safely, use `pkill -u $USER -f main.py`.

Use `nohup` to run these in the background across multiple GPUs:

```bash
# Regime A: Fixed Opponent (Base Policy)
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime A --iterations 2000 --seed 42 --llm-gpu 0 > run_regime_A_42.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime A --iterations 2000 --seed 43 --llm-gpu 1 > run_regime_A_43.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime A --iterations 2000 --seed 44 --llm-gpu 2 > run_regime_A_44.log 2>&1 &

# Regime B: Evolving Opponent (Current Policy)
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime B --iterations 2000 --seed 42 --llm-gpu 0 > run_regime_B_42.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime B --iterations 2000 --seed 43 --llm-gpu 1 > run_regime_B_43.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime B --iterations 2000 --seed 44 --llm-gpu 2 > run_regime_B_44.log 2>&1 &

# Regime C: Evolving Opponent + High Replay (50k buffer)
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime C --iterations 2000 --seed 42 --llm-gpu 0 > run_regime_C_42.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime C --iterations 2000 --seed 43 --llm-gpu 1 > run_regime_C_43.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime C --iterations 2000 --seed 44 --llm-gpu 2 > run_regime_C_44.log 2>&1 &

# Regime D: Historical Opponent Pool (League / FSP)
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime D --iterations 2000 --seed 42 --llm-gpu 0 > run_regime_D_42.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime D --iterations 2000 --seed 43 --llm-gpu 1 > run_regime_D_43.log 2>&1 &
nohup conda run --no-capture-output -n rl_sim python -u main.py --algo DAPO --backend openspiel --env kuhn_poker --regime D --iterations 2000 --seed 44 --llm-gpu 2 > run_regime_D_44.log 2>&1 &
```

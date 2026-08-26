# Generalized LLM Multi-Agent RL (MARL) Framework

A scalable, multi-algorithm reinforcement learning framework for training large language models (LLMs) like Qwen via self-play in diverse multi-agent environments. 

This branch (`opponent-regimes`) explores experimental self-play regimes designed to measure representation collapse in multi-agent learning environments.

## Features

* **LLM Environment Adapters**: Bridges numerical arrays to text prompts for LLMs using the `[ACTION: <move>]` grammar.
  * **TextArena**: Tic-Tac-Toe, Kuhn Poker, Negotiation, etc.
  * **OpenSpiel**: Leduc Poker, Hanabi, Kuhn Poker (handles chance nodes natively).
  * **PettingZoo**: Classic AEC games like Tic-Tac-Toe, Connect Four, Simple Spread.
  * **JaxMARL**: GPU-accelerated environments like Overcooked and SMAX.
* **White-Box RL Optimization**: Actual PyTorch sequence log-probability gradient computations.
  * `REINFORCE`: Standard policy gradient with running baseline.
  * `REINFORCE++`: TRR++ logic for advantage boosting in proposer-solver setups.
  * `DAPO`: PPO-style surrogate clipping with entropy preservation.
  * `GRPO`: Group Relative Policy Optimization without a value network.
* **Hardware Partitioning**: Dynamically maps the LLM (`Qwen/Qwen2.5-0.5B-Instruct`) to GPU 0 and environments (like JaxMARL) to GPU 1 to prevent VRAM overflow.

## 🛠️ Installation

We provide an automated setup script (`install.sh`) that works on both macOS (MPS) and Linux Servers (CUDA 12.1).

```bash
git clone https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play.git
cd qwen_self_play
git checkout opponent-regimes

chmod +x install.sh
./install.sh
```

## 🧪 Health Check

Before initiating a distributed training run, verify all C++ and JAX dependencies are working properly:
```bash
conda activate rl_sim
python health_check.py
```

## 🎮 Usage

You can run the full dual-GPU training loop via the `main.py` entrypoint. The script dynamically routes the selected environment and algorithm.

```bash
# Test GRPO on TextArena Tic-Tac-Toe
python main.py --algo GRPO --backend textarena --env TicTacToe-v0

# Test REINFORCE on PettingZoo
python main.py --algo REINFORCE --backend pettingzoo --env tictactoe_v3

# Test REINFORCE++ on OpenSpiel Kuhn Poker
python main.py --algo REINFORCE_PLUS --backend openspiel --env kuhn_poker

# Test DAPO on JaxMARL Overcooked
python main.py --algo DAPO --backend jaxmarl --env overcooked
```

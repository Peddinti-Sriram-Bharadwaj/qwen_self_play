# Qwen Continual Self-Play (MARL Framework)

Welcome to the Generalized LLM Multi-Agent Reinforcement Learning framework. This repository is dedicated to researching how Large Language Models (LLMs), such as Qwen, evolve during continual Reinforcement Learning (RL) via Self-Play.

We are specifically investigating **Plasticity Collapse** (the degradation of a neural network's ability to learn new things after prolonged RL fine-tuning) and how **Chain-of-Thought (CoT)** reasoning might mitigate it.

## 🌿 Branch Navigation

Active development and specific experiments are tracked in specialized branches, rather than `master`. 

Please switch to the relevant branch for your research:

* **[multi-algo-comparison](https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play/tree/multi-algo-comparison)**
  * **Status:** 🟢 Active Development / Stable
  * **Contents:** The cutting-edge unified MARL framework. Contains cross-framework adapters for `TextArena`, `OpenSpiel`, `PettingZoo`, and `JaxMARL`. Implements actual PyTorch white-box RL optimization for `REINFORCE`, `REINFORCE++`, `DAPO`, and `GRPO`. Supports dual-GPU hardware partitioning for heavy models.

* **[cot-plasticity-research](https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play/tree/cot-plasticity-research)**
  * **Status:** 🟡 Experimental / Baseline
  * **Contents:** The foundational scaffolding for measuring plasticity collapse. Contains the core logic for extracting CoT latents, tracking feature variance, and calculating dormant neuron percentages during the batched generation loop.

## ⚙️ Quick Start

To use the framework, clone this repository and switch to the `multi-algo-comparison` branch, then run the automated setup script on your local or remote server:

```bash
git clone https://github.com/Peddinti-Sriram-Bharadwaj/qwen_self_play.git
cd qwen_self_play

# Switch to the active framework branch
git checkout multi-algo-comparison

# Install all dependencies (Conda, PyTorch CUDA 12.1, and MARL Environments)
chmod +x install.sh
./install.sh
```

## 📚 Acknowledgements
Inspired by approaches detailed in SPIRAL, SCO-PAL, and modern RLHF paradigms.

# Continual RL for LLMs: CoT Plasticity Research

This branch (`cot-plasticity-research`) contains the foundational scaffolding for measuring plasticity collapse and feature variance in Large Language Models (LLMs) during continual Reinforcement Learning from Human Feedback (RLHF) / Self-Play.

It serves as the base architectural layer for the multi-algorithm and multi-environment comparisons built on top of it.

## 🎯 Goal
Investigate whether models like Qwen experience "plasticity collapse" (the inability to learn new tasks) when trained continuously via self-play, and explore if Chain-of-Thought (CoT) reasoning acts as a buffer against this collapse.

## 🏗️ Architecture

- **`agent.py`**: Wraps the Hugging Face `Qwen2.5-0.5B-Instruct` model and manages tokenization, batched generation, and extraction of hidden state latents (CoT Latents).
- **`self_play.py`**: Manages the batched interaction loop between the Agent and the environments.
- **`main.py`**: The central orchestrator that initializes the models, the algorithm strategies, and the datasets.
- **`trainers/`**: Houses the abstract `TrainerStrategy` interfaces for different algorithms (e.g., PPO, GRPO).

## 📊 Evaluation
- **`evaluate_plasticity.py`**: A dedicated script to parse training logs and plot metrics like "Feature Variance" and "Dormant Neuron Percentage" to visually quantify plasticity collapse.

## 🔀 Branches
* **`multi-algo-comparison`**: (Active Development) Introduces `EnvFactory` wrappers for TextArena, OpenSpiel, PettingZoo, and JaxMARL, along with white-box PyTorch optimization for algorithms like `REINFORCE`, `REINFORCE++`, and `DAPO`.

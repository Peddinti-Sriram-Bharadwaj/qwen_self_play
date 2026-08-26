# Continual RL for LLMs: CoT Plasticity Research

This branch (`cot-plasticity-research`) contains the baseline implementation for measuring plasticity collapse and feature variance in Large Language Models (LLMs) during continual reinforcement learning and self-play.

It serves as the foundational architectural layer for multi-algorithm and multi-environment comparisons.

## Goal
Investigate whether models like Qwen experience plasticity collapse (the degradation of the ability to learn new tasks) when trained continuously via self-play, and explore if Chain-of-Thought (CoT) reasoning acts as a stabilizing mechanism against this effect.

## Architecture

- **`agent.py`**: Wraps the Hugging Face `Qwen2.5-0.5B-Instruct` model and manages tokenization, batched generation, and extraction of hidden state latents (CoT Latents).
- **`self_play.py`**: Manages the batched interaction loop between the Agent and the environments.
- **`main.py`**: The central orchestrator that initializes the models, algorithm strategies, and datasets.
- **`trainers/`**: Houses the abstract `TrainerStrategy` interfaces for different algorithms (e.g., PPO, GRPO).

## Evaluation
- **`evaluate_plasticity.py`**: A dedicated script to parse training logs and plot metrics, such as Feature Variance and Dormant Neuron Percentage, to quantify plasticity collapse over time.

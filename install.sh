#!/bin/bash
set -e

echo "==============================================="
echo "   Self-Play Continual Qwen Setup Script"
echo "==============================================="

# 1. Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 2. Create the environment
ENV_NAME="rl_sim"
echo "Creating Conda environment: $ENV_NAME..."
conda create -n $ENV_NAME python=3.10 -y

# 3. Source conda to activate it within the script
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 4. Install CUDA PyTorch (standard for Linux/A1000/A6000 servers)
echo "Installing PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Python dependencies from requirements.txt
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "==============================================="
echo "Setup Complete!"
echo "Run: 'conda activate rl_sim' to get started."
echo "You can verify your installation by running: 'python health_check.py'"
echo "==============================================="

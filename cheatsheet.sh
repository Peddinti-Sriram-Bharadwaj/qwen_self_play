# Server Management & Monitoring
# ------------------------------
# Check your own CPU/Memory usage (useful on shared servers)
htop -u $USER

# Check running background jobs associated with Python
ps aux | grep python

# Kill all your running python scripts (Emergency Stop)
pkill -u $USER -f main.py

# Check NVIDIA GPU Usage (The standard view)
watch -n 1 nvidia-smi

# Check exactly how much Free Memory is available across all GPUs
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

# Check logs of a running experiment in real-time
tail -f run_targeted_A.log
# Check overall disk space and free space on the current partition
df -h ~

# Check strict storage quotas assigned to your user ID
quota -s

# Check total size of your home directory
du -sh ~


# Running Targeted Experiments (ICLR Sprint)
# ------------------------------------------
# Make sure you are in the conda environment first
# conda activate rl_sim

# Fallback Plan: Run Leduc Poker (More complex than Kuhn Poker) to force plasticity collapse
# We also increase Learning Rate from 1e-5 to 1e-4 in trainer.py if needed, but try Leduc first.

# 1. Regime A (Control - Fixed Opponent)
nohup python -u main.py --algo DAPO --backend openspiel --env leduc_poker --regime A --iterations 2000 --seed 42 --lr 1e-4 --beta 0.0 --llm-gpu 0 > run_targeted_A.log 2>&1 &

# 2. Regime B (Self-Play Collapse)
nohup python -u main.py --algo DAPO --backend openspiel --env leduc_poker --regime B --iterations 2000 --seed 42 --lr 1e-4 --beta 0.0 --llm-gpu 0 > run_targeted_B.log 2>&1 &

# 3. Regime D (League Training / Fictitious Play Fix)
nohup python -u main.py --algo DAPO --backend openspiel --env leduc_poker --regime D --iterations 2000 --seed 42 --lr 1e-4 --beta 0.0 --llm-gpu 0 > run_targeted_D.log 2>&1 &

# 4. Optional: Regime C (High Replay Buffer)
nohup python -u main.py --algo DAPO --backend openspiel --env leduc_poker --regime C --iterations 2000 --seed 42 --lr 1e-4 --beta 0.0 --llm-gpu 0 > run_targeted_C.log 2>&1 &

# Git Management (Syncing Branches)
# ----------------------------------
# See which branch you are currently on, and if there are uncommitted changes
git status

# Fetch the latest information about all remote branches without merging
git fetch --all

# Switch to and pull the main branch
git checkout main
git pull origin main

# Switch to and pull the experimental harness branch (Full Grid)
git checkout experimental-harness
git pull origin experimental-harness

# Switch to and pull the targeted runs branch (ICLR Sprint)
git checkout targeted-runs
git pull origin targeted-runs

# Discard all local changes in the current branch (WARNING: destroys uncommitted work)
# git reset --hard HEAD
# git clean -fd

# See a quick view of the last 5 commits
git log --oneline -n 5


# Running Anchored Self-Play (Code Repair)
# ----------------------------------------
# 1. Standard Run (High K for smoother signal) on GPU 1
# nohup python -u code_self_play.py --gpu 1 --beta 0.04 --K 16 --run-name high_k_anchored > run_high_k.log 2>&1 &

# 2. Unanchored Collapse Experiment (Beta = 0.0) on GPU 1
# nohup python -u code_self_play.py --gpu 1 --beta 0.0 --K 4 --run-name unanchored_collapse > run_unanchored.log 2>&1 &

# 3. High Generator Variance (High G, High K) on GPU 2
# nohup python -u code_self_play.py --gpu 2 --beta 0.04 --K 16 --G 16 --run-name high_g_high_k_anchored > run_high_g.log 2>&1 &

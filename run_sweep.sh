#!/bin/bash
# run_sweep.sh
# This script runs a full multi-seed variance study across 2 GPUs sequentially.
# It ensures strict isolation of checkpoints and no overwriting.

BETA=0.01
MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"

run_sweep_for_lr() {
    local LR=$1
    local BASE_CKPT_DIR="checkpoints_sweep_lr${LR}"
    
    echo "=========================================================="
    echo "Starting Variance Sweep for LR: $LR, BETA: $BETA"
    echo "=========================================================="

    # ----------------------------------------------------
    # GPU 0: Assigned Seeds 42, 43, 44
    # ----------------------------------------------------
    (
        for SEED in 42 43 44; do
            echo "[GPU 0] Starting Seed $SEED for LR $LR..."
            python -u main_experiment.py \
                --gpu 0 \
                --model_name $MODEL \
                --full_finetune \
                --lr $LR \
                --beta $BETA \
                --seed $SEED \
                --run-name "sweep_0.5B_lr${LR}_seed${SEED}" \
                --ckpt_dir "$BASE_CKPT_DIR/seed_$SEED" \
                --data_dir data \
                > "run_gpu0_lr${LR}_seed${SEED}.log" 2>&1
            echo "[GPU 0] Finished Seed $SEED for LR $LR."
        done
    ) &

    # ----------------------------------------------------
    # GPU 1: Assigned Seeds 45, 46
    # ----------------------------------------------------
    (
        for SEED in 45 46; do
            echo "[GPU 1] Starting Seed $SEED for LR $LR..."
            python -u main_experiment.py \
                --gpu 1 \
                --model_name $MODEL \
                --full_finetune \
                --lr $LR \
                --beta $BETA \
                --seed $SEED \
                --run-name "sweep_0.5B_lr${LR}_seed${SEED}" \
                --ckpt_dir "$BASE_CKPT_DIR/seed_$SEED" \
                --data_dir data \
                > "run_gpu1_lr${LR}_seed${SEED}.log" 2>&1
            echo "[GPU 1] Finished Seed $SEED for LR $LR."
        done
    ) &

    # Wait for both GPUs to finish all their assigned seeds before proceeding
    wait
    echo "Sweep for LR $LR is complete!"
}

# 1. Run the conservative baseline sweep (1e-5)
run_sweep_for_lr 1e-5

# 2. Run the aggressive collapse sweep (1e-4)
run_sweep_for_lr 1e-4

echo "=========================================================="
echo "ALL SWEEPS COMPLETE!"
echo "=========================================================="

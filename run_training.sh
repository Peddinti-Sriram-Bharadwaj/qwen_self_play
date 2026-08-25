#!/bin/bash
ALGO=${1:-PPO}
echo "Starting continuous $ALGO training loop in the background..."
echo "Logs will be written to training_$ALGO.log"
nohup accelerate launch --multi_gpu main.py --algo $ALGO > training_$ALGO.log 2>&1 &
echo "Process started with PID: $!"
echo "To view logs, run: tail -f training_$ALGO.log"

#!/bin/bash
echo "Starting continuous training loop in the background..."
echo "Logs will be written to training.log"
nohup python main.py > training.log 2>&1 &
echo "Process started with PID: $!"
echo "To view logs, run: tail -f training.log"

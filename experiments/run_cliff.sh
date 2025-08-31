#!/usr/bin/env bash
set -e
python rl/train.py --env cliff --algo qlearning,sarsa --episodes 1000 --alpha 0.1 --gamma 0.99 --eps-start 1.0 --eps-end 0.05 --ma-window 50

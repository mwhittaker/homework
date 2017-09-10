#! /usr/bin/env bash

set -euo pipefail

main() {
    python behavioral_cloning.py \
        --batch_size 1000 \
        --training_steps 1000 \
        --checkpoint_dir bc_checkpoints/Reacher-v1 \
        expert_rollouts/Reacher-v1_50000rollouts.pkl
    python run_trained.py \
        --num_rollouts 1000 \
        --stats_file bc_rollouts/Reacher-v1_1000rollouts.json \
        bc_checkpoints/Reacher-v1/Reacher-v1-29.meta \
        Reacher-v1
}

main

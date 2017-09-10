#! /usr/bin/env bash

set -euo pipefail

main() {
    python behavioral_cloning.py \
        --checkpoint_dir bc_checkpoints/Reacher-v1 \
        expert_rollouts/Reacher-v1_10rollouts.pkl
    python run_trained.py \
        --num_rollouts 20 \
        --stats_file bc_rollouts/Reacher-v1_20rollouts.json \
        bc_checkpoints/Reacher-v1/Reacher-v1-901.meta \
        Reacher-v1
}

main

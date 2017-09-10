#! /usr/bin/env bash

set -euo pipefail

main() {
    # python behavioral_cloning.py \
        # --batch_size 1000 \
        # --training_steps 1000 \
        # --checkpoint_dir bc_checkpoints/Reacher-v1 \
        # expert_rollouts/Reacher-v1_50000rollouts.pkl
    # python run_trained.py \
        # --num_rollouts 1000 \
        # --stats_file bc_rollouts/Reacher-v1_1000rollouts.json \
        # bc_checkpoints/Reacher-v1/Reacher-v1-29.meta \
        # Reacher-v1

    # python behavioral_cloning.py \
        # --batch_size 250 \
        # --training_steps 1000000 \
        # --hidden1 32 \
        # --hidden2 32 \
        # --checkpoint_dir bc_checkpoints/HalfCheetah-v1-32x32 \
        # expert_rollouts/HalfCheetah-v1_1500rollouts.pkl
    # python run_trained.py \
        # --num_rollouts 10 \
        # --stats_file bc_rollouts/HalfCheetah-v1_1000rollouts.json \
        # bc_checkpoints/HalfCheetah-v1-32x32/HalfCheetah-v1-32x32-1.meta \
        # HalfCheetah-v1

    time python behavioral_cloning.py \
        --batch_size 995 \
        --training_steps 1000000 \
        --hidden1 32 \
        --hidden2 32 \
        --checkpoint_dir bc_checkpoints/Walker2d-v1-32x32 \
        expert_rollouts/Walker2d-v1_1000rollouts.pkl
    python run_trained.py \
        --num_rollouts 100 \
        --stats_file bc_rollouts/Walker2d-v1_100rollouts.json \
        bc_checkpoints/Walker2d-v1-32x32/Walker2d-v1-32x32-34001.meta \
        Walker2d-v1
}

main

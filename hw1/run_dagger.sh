#! /usr/bin/env bash

main() {
    python dagger.py \
        --num_rollouts 100 \
        --num_iterations 10 \
        --hidden1 32 \
        --hidden2 32 \
        --batch_size 10000 \
        --training_steps 5000 \
        --stats_file dagger_rollouts/Walker2d-v1.json \
        --rollouts_file expert_rollouts/Walker2d-v1_1000rollouts.pkl \
        --checkpoint_dir bc_checkpoints/Walker2d-v1-32x32 \
        experts/Walker2d-v1.pkl
}

main

#! /usr/bin/env bash

set -euo pipefail

main() {
    for i in 50 100 250 500 750 1000 1250 1500; do
        time python behavioral_cloning.py \
            --batch_size 1000 \
            --training_steps $((i * 50)) \
            --hidden1 16 \
            --hidden2 16 \
            --checkpoint_dir "bc_checkpoints/HalfCheetah-v1-16x16-${i}rollouts" \
            "expert_rollouts/cheetah_splits/HalfCheetah-v1_${i}rollouts.pkl"
        meta_file=$(ls bc_checkpoints/HalfCheetah-v1-16x16-${i}rollouts/*.meta | head -n 1)
        time python run_trained.py \
            --num_rollouts 50 \
            --stats_file bc_rollouts/HalfCheetah-v1-${i}rollouts_50rollouts.json \
            "$meta_file" \
            HalfCheetah-v1
    done
}

main

#! /usr/bin/env bash

set -euo pipefail

main() {
    experts=()
    experts+=(Ant-v1)
    experts+=(HalfCheetah-v1)
    experts+=(Hopper-v1)
    experts+=(Humanoid-v1)
    experts+=(Reacher-v1)
    experts+=(Walker2d-v1)

    readonly num_rollouts=10
    readonly data_dir=rollouts

    for expert in "${experts[@]}"; do
        stats_file="${data_dir}/${expert}_${num_rollouts}rollouts.json"
        rollouts_file="${data_dir}/${expert}_${num_rollouts}rollouts.pkl"
        if [[ -f "$stats_file" ]]; then
            echo "$stats_file already exists. Not running $expert."
        fi
        if [[ -f "$rollouts_file" ]]; then
            echo "$rollouts_file already exists. Not running $expert."
        fi

        set -x
        python run_expert.py \
            --num_rollouts="$num_rollouts" \
            --stats_file="$stats_file" \
            --rollouts_file="$rollouts_file" \
            "experts/$expert.pkl" \
            "$expert"
        set +x
    done
}

main

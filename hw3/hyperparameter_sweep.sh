#! /usr/bin/env bash

set -euo pipefail

main() {
    set -x
    for freq in 1000 5000 20000; do
        python run_dqn_atari.py \
            --num_timesteps 6000000 \
            --target_update_freq $freq \
            --checkpoint_dir "checkpoints/target_update_freq_$freq"
    done
    set +x
}

main

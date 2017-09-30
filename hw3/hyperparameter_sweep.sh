#! /usr/bin/env bash

set -euo pipefail

main() {
    set -x
    for bufsize in 1000 10000 100000; do
        python run_dqn_atari.py \
            --num_timesteps 6000000 \
            --replay_buffer_size $bufsize \
            --checkpoint_dir "checkpoints/replay_buffer_size_$bufsize"
    done
    set +x
}

main

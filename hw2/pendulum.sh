#! /usr/bin/env bash

set -euo pipefail

main() {
    python train_pg.py \
        InvertedPendulum-v1 \
        --verbose \
        --n_layers 1 \
        --size 4 \
        -n 100 \
        -b 2000 \
        -e 1 \
        --discount 0.99 \
        --learning_rate 0.005 \
        -rtg \
        --exp_name pendulum_lb_rtg_na
}

main

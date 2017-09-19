#! /usr/bin/env bash

set -euo pipefail

main() {
    python train_pg.py \
        InvertedPendulum-v1 \
        --verbose \
        --n_layers 2 \
        --size 32 \
        --seed 2 \
        -n 100 \
        -b 10000 \
        -e 2 \
        --discount 1 \
        --learning_rate 0.01 \
        -rtg \
        --nn_baseline \
        --exp_name pendulum_lb_rtg_na_baseline
}

main

#! /usr/bin/env bash

set -euo pipefail

main() {
    fixed_flags="-ep 150 --discount 0.9 -n 100"
    python train_pg.py \
        HalfCheetah-v1 \
        $fixed_flags \
        --verbose \
        --n_layers 2 \
        --size 32 \
        --seed 3 \
        -b 50000 \
        -e 1 \
        --learning_rate 0.02 \
        -rtg \
        --exp_name cheetah
}

main

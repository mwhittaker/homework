#! /usr/bin/env bash

set -euo pipefail

main() {
    flags="--verbose --n_layers 1 --size 32"
    set -x
    python train_pg.py CartPole-v0 $flags -n 100 -b 1000 -e 5      -dna --exp_name sb_no_rtg_dna
    python train_pg.py CartPole-v0 $flags -n 100 -b 1000 -e 5 -rtg -dna --exp_name sb_rtg_dna
    python train_pg.py CartPole-v0 $flags -n 100 -b 1000 -e 5 -rtg      --exp_name sb_rtg_na
    python train_pg.py CartPole-v0 $flags -n 100 -b 5000 -e 5      -dna --exp_name lb_no_rtg_dna
    python train_pg.py CartPole-v0 $flags -n 100 -b 5000 -e 5 -rtg -dna --exp_name lb_rtg_dna
    python train_pg.py CartPole-v0 $flags -n 100 -b 5000 -e 5 -rtg      --exp_name lb_rtg_na
    set +x
}

main

#! /usr/bin/env bash

set -euo pipefail

main() {
    n=100
    e=5
    flags="--verbose --n_layers 1 --size 32"
    set -x
    python train_pg.py CartPole-v0 $flags -n $n -b 1000 -e $e      -dna --exp_name sb_no_rtg_dna
    python train_pg.py CartPole-v0 $flags -n $n -b 1000 -e $e -rtg -dna --exp_name sb_rtg_dna
    python train_pg.py CartPole-v0 $flags -n $n -b 1000 -e $e -rtg      --exp_name sb_rtg_na
    python train_pg.py CartPole-v0 $flags -n $n -b 5000 -e $e      -dna --exp_name lb_no_rtg_dna
    python train_pg.py CartPole-v0 $flags -n $n -b 5000 -e $e -rtg -dna --exp_name lb_rtg_dna
    python train_pg.py CartPole-v0 $flags -n $n -b 5000 -e $e -rtg      --exp_name lb_rtg_na
    set +x
}

main

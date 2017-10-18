#! /usr/bin/env bash

set -euo pipefail

main() {
    python main.py -v --onpol_iters 15 --seed 9001
}

main

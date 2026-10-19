#!/usr/bin/env bash
# FixMatch baseline vs EMC-adaptive threshold, FI, three label budgets, three seeds.
set -euo pipefail
for n in 80 800 1600; do
  for rule in fixed ratio; do
    for seed in 0 1 2; do
      python -m ecc.ssl --config configs/ssl_fi.yaml \
        --set ssl.n_labeled=$n ssl.rule=$rule seed=$seed output_dir=runs/ssl_fi_n${n}_${rule}_s${seed}
    done
  done
done

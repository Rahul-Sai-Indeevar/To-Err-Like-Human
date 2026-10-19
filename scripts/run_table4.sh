#!/usr/bin/env bash
# Reproduce paper Table 4 (FI): alpha in {0, 0.2, 1.0} x backbone in {resnet18, resnet50, resnet101}
set -euo pipefail
for net in resnet18 resnet50 resnet101; do
  for a in 0.0 0.2 1.0; do
    python -m ecc.supervised --config configs/fi_resnet50.yaml \
      --set model.name=$net loss.alpha=$a output_dir=runs/fi_${net}_a${a}
  done
done

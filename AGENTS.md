# AGENTS.md — guidance for coding agents (opencode etc.)

Project: re-implementation of the NeurIPS 2024 paper "To Err Like Human" (ECC/EMC metrics for visual
emotion recognition). See README.md for layout and the table of unverified assumptions.

## Commands
- Install: `pip install -e ".[dev]"`
- Test (no GPU/data needed): `pytest -q`
- Train: `python -m ecc.supervised --config configs/fi_resnet50.yaml --set key=value ...`
- Semi-supervised: `python -m ecc.ssl --config configs/ssl_fi.yaml --set key=value ...`

## Conventions
- Code lives in `src/ecc/`; experiments are configured via YAML in `configs/`, overridden with `--set a.b=c`.
- Metrics are computed from a confusion matrix (rows = true, cols = predicted); keep them in `metrics.py`.
- `emotion.WHEEL_ORDER` is the single source of truth for the emotion wheel.
- Never touch the test split for model selection; use val.
- New ideas go behind a config flag and must not change the default (paper) behaviour.
- Add/extend tests in `tests/` for any change to metrics, losses or thresholds; keep `pytest -q` green.
- Do not commit `data/`, `runs/` or checkpoints (see .gitignore).

## Open TODOs (highest value first)
1. Verify `WHEEL_ORDER`, EMC weight and Eq. 5 against the official repo's `metric.py`/`train.py`.
2. Reproduce Table 4 (FI, ResNet50) and compare numbers in README.
3. Port EMC-adaptive threshold into S2-VER for the Table 1 lower block.

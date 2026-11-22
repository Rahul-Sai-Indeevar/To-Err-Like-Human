"""Replay all 12 completed full-data checkpoints, sequentially on the same GPU.

This is an inference audit, not a new training experiment. It must run after the
training process exits; the shared GPU lock rejects an overlapping launch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from torchvision.models import resnet18


METRICS = ("acc", "acc2", "ecc", "emc", "gsee", "brier", "dirichlet", "error_energy")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_replay(saved_probabilities, saved_labels, replay_probabilities, replay_labels,
                   saved_metrics, replay_metrics, tolerance=1e-6):
    """CPU-only comparison; return a structured failure rather than hiding it."""
    a, b = np.asarray(saved_probabilities), np.asarray(replay_probabilities)
    y, z = np.asarray(saved_labels), np.asarray(replay_labels)
    failures = []
    same_shape = a.shape == b.shape and a.ndim == 2 and y.shape == z.shape == (len(a),)
    if not same_shape:
        failures.append("probability_or_label_shape")
    labels_match = bool(np.array_equal(y, z))
    if not labels_match:
        failures.append("label_identity_or_order")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    valid = finite and a.ndim == b.ndim == 2 and all(
        np.all(p >= 0) and np.all(p <= 1) and np.allclose(p.sum(1), 1, atol=2e-6, rtol=0)
        for p in (a, b)
    )
    if not valid:
        failures.append("invalid_probabilities")
    max_difference = float(np.max(np.abs(a - b))) if same_shape and finite and len(a) else None
    agreement = float(np.mean(a.argmax(1) == b.argmax(1))) if same_shape and valid and len(a) else None
    if max_difference is None or max_difference > tolerance:
        failures.append("probability_tolerance")
    if agreement != 1.0:
        failures.append("argmax_agreement")
    metric_checks = {}
    for name in METRICS:
        if name not in saved_metrics or name not in replay_metrics:
            metric_checks[name] = {"passed": False, "max_absolute_difference": None}
        else:
            expected, actual = np.asarray(saved_metrics[name]), np.asarray(replay_metrics[name])
            comparable = expected.shape == actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all()
            difference = float(np.max(np.abs(expected - actual))) if comparable else None
            metric_checks[name] = {"passed": bool(difference is not None and difference <= tolerance),
                                   "max_absolute_difference": difference}
        if not metric_checks[name]["passed"]:
            failures.append(f"metric_{name}")
    return {"passed": not failures, "n_images": int(len(y)), "labels_match": labels_match,
            "probabilities_valid": bool(valid), "max_probability_difference": max_difference,
            "argmax_agreement": agreement, "metric_checks": metric_checks, "failures": failures}


def preflight(study, repo):
    """No GPU queries or inference: establish completeness and pinned source first."""
    protocol = read_json(study / "protocol.json")
    unsigned = dict(protocol)
    expected = unsigned.pop("protocol_hash")
    actual = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != expected:
        raise ValueError("Protocol canonical hash mismatch")
    seeds, methods = protocol["args"]["seeds"], protocol["args"]["methods"]
    if seeds != list(range(3, 9)) or methods != ["ce", "graph"] or protocol["args"]["epochs"] != 8 or protocol["args"]["batch_size"] != 64:
        raise ValueError("Expected the locked six-seed-pair, eight-epoch, batch64 full-data protocol")
    if protocol["args"]["skip_test"]:
        raise ValueError("A benchmark without test predictions cannot be replayed as the full study")
    sources = {"runner": ("runner_source.py", "scripts/run_full_seed_evidence.py"),
               "ecc.graph": ("graph_source.py", "src/ecc/graph.py"),
               "ecc.metrics": ("metrics_source.py", "src/ecc/metrics.py"),
               "ecc.emotion": ("emotion_source.py", "src/ecc/emotion.py")}
    for name, (snapshot, current) in sources.items():
        for path in (study / snapshot, repo / current):
            if sha256(path) != protocol["sources"][name]:
                raise ValueError(f"Pinned source hash mismatch: {path}")
    runs = []
    for seed in seeds:
        for method in methods:
            dest = study / f"{method}_seed{seed}"
            for name in ("result.json", "history.json", "best.pt", "val_predictions.npz", "predictions.npz"):
                if not (dest / name).is_file():
                    raise FileNotFoundError(f"Full training is incomplete; missing {dest / name}")
            result = read_json(dest / "result.json")
            history = read_json(dest / "history.json")
            if result["protocol_hash"] != expected or result["method"] != method or result["seed"] != seed:
                raise ValueError(f"Run identity mismatch: {dest}")
            if [h["epoch"] for h in history] != list(range(1, 9)):
                raise ValueError(f"Run has not completed all eight epochs: {dest}")
            best = max(range(len(history)), key=lambda i: history[i]["val"]["ecc"])
            if result["best_epoch"] != best + 1 or result["best_validation_ecc"] != history[best]["val"]["ecc"]:
                raise ValueError(f"Validation checkpoint-selection mismatch: {dest}")
            runs.append((dest, result))
    for seed in seeds:
        pair = [r for _, r in runs if r["seed"] == seed]
        for key in ("initial_state_sha256", "augmentation_plan_sha256"):
            if len({r[key] for r in pair}) != 1:
                raise ValueError(f"Paired {key} mismatch for seed{seed}")
    return protocol, runs


def run(args):
    repo = args.repo.resolve()
    study = args.study.resolve() if args.study.is_absolute() else (repo / args.study).resolve()
    protocol, runs = preflight(study, repo)
    spec = importlib.util.spec_from_file_location("frozen_full_seed_runner", study / "runner_source.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    for name in ("ecc.graph", "ecc.metrics", "ecc.emotion"):
        actual_module = importlib.import_module(name)
        if sha256(actual_module.__file__) != protocol["sources"][name]:
            raise ValueError(f"Imported module differs from pinned source: {name}")
    output = args.output.resolve() if args.output else study / "checkpoint_replay.json"
    lock_file = Path(protocol["args"]["lock_file"])
    if not lock_file.is_absolute():
        lock_file = repo / lock_file
    # This lock is acquired before any CUDA query/allocation. A training process
    # with its own final result already written can still hold it while exiting.
    with frozen.gpu_lock(lock_file):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Same-host replay requires the single CUDA GPU used for training")
        device = torch.device("cuda")
        runtime = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                   "numpy": np.__version__, "python": platform.python_version(),
                   "gpu_name": torch.cuda.get_device_name(device)}
        for key, value in runtime.items():
            if value != protocol[key]:
                raise ValueError(f"Same-host replay runtime changed: {key}: {value} vs {protocol[key]}")
        torch.set_num_threads(3)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        root = Path(protocol["args"]["data_root"])
        root = root if root.is_absolute() else repo / root
        if sha256(root / "manifest.json") != protocol["data_manifest_sha256"]:
            raise ValueError("Dataset preparation manifest changed")
        datasets, metadata = frozen.load_dataset(root, protocol["classes"], frozen.SPLITS)
        if metadata != protocol["datasets"]:
            raise ValueError("CSV identities, class counts or input cache hashes changed")
        images = frozen.Images(datasets, device)
        W = frozen.build_cost_matrix(protocol["classes"])
        polarity = frozen.polarity_vector(protocol["classes"])
        report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                  "protocol_hash": protocol["protocol_hash"], "runtime": runtime,
                  "expected_models": 12, "completed_models": 0, "tolerance": args.tolerance,
                  "required_argmax_agreement": 1.0, "amp": not protocol["args"]["no_amp"],
                  "batch_size": 64, "input_metadata_reverified": True, "models": []}
        frozen.atomic_json(output, report)
        for dest, result in runs:
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(W))
            model.load_state_dict(torch.load(dest / "best.pt", map_location="cpu"), strict=True)
            model.to(device)
            record = {"method": result["method"], "seed": result["seed"], "splits": {}}
            for split, filename in (("val", "val_predictions.npz"), ("test", "predictions.npz")):
                p, y = frozen.predict(model, images, split, 64, report["amp"])
                with np.load(dest / filename, allow_pickle=False) as saved:
                    check = compare_replay(saved["probabilities"], saved["labels"], p, y,
                                           result[split], frozen.evaluate(p, y, W, polarity), args.tolerance)
                record["splits"][split] = check
            record["passed"] = all(check["passed"] for check in record["splits"].values())
            report["models"].append(record)
            report["completed_models"] += 1
            report["status"] = "running" if record["passed"] else "failed"
            frozen.atomic_json(output, report)
            print("REPLAY", dest.name, "PASS" if record["passed"] else "FAIL", flush=True)
            del model
            torch.cuda.empty_cache()
            if not record["passed"]:
                raise RuntimeError(f"Checkpoint replay failed; diagnostics preserved in {output}")
        report["status"] = "passed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        frozen.atomic_json(output, report)
        print("COMPLETE: all12 checkpoint val/test replays passed", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--study", type=Path, default=Path("outputs/full_seed_replication"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("Tolerance must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()

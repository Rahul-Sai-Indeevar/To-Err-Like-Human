"""Train the paired Haar and RGB control models used on slide 19.

Every model uses the locked paired-seed protocol.
All models train before any new test predictions are generated. A killed partial
model is retrained from its recorded initial state; completed models are verified
and skipped. Checkpoints do not claim exact mid-epoch continuation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import yaml

from ecc import experiment_data as shared
from ecc.experiment_data import file_sha
from ecc.seed_statistics import paired_seed_summary
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.wavelets import WaveletEmotionModel

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_wavelet_study.py", "src/ecc/experiment_data.py",
    "src/ecc/wavelets.py", "src/ecc/emotion.py", "src/ecc/metrics.py", "src/ecc/seed_statistics.py",
)
METHODS = ("wavelet_fusion", "rgb_dual")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    if tuple(config["methods"]) != METHODS:
        raise ValueError("Expected methods: wavelet_fusion, rgb_dual")
    seeds = config["seeds"]
    if len(seeds) != len(set(seeds)) or not seeds or any(type(s) is not int for s in seeds):
        raise ValueError("Seeds must be unique integers")
    if config["epochs"] < 1 or config["batch_size"] < 2 or config["levels"] != 2:
        raise ValueError("Positive epochs, batch>=2, and the locked J=2 partition are required")
    return config


def make_initial(seed, nclasses):
    shared.seed_everything(seed)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, nclasses)
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_runtime(device_name):
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def initialize_protocol(name, data_root, out, config, config_path, device):
    classes = (data_root / "classes.txt").read_text(encoding="utf8").splitlines()
    datasets, metadata = shared.load_dataset(data_root, classes, shared.SPLITS)
    plan_path = ROOT / "docs/wavelet_study/STUDY_PLAN.md"
    sources = {p: file_sha(ROOT / p) for p in SOURCE_FILES}
    args = {"data_root": str(data_root.resolve()), "epochs": config["epochs"],
            "seeds": config["seeds"], "methods": config["methods"],
            "batch_size": config["batch_size"], "levels": config["levels"]}
    protocol = {
        "dataset": name, "args": args, "classes": classes, "datasets": metadata,
        "sources": sources,
        "data_manifest_sha256": file_sha(data_root / "manifest.json") if (data_root / "manifest.json").exists() else None,
        "config_sha256": file_sha(config_path), "study_plan_path": str(plan_path.relative_to(ROOT)),
        "study_plan_sha256": file_sha(plan_path),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "device": device.type,
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "loss": "eight-class cross entropy only, identical for all methods",
        "optimizer": {"name": "AdamW", "lr": .0001, "weight_decay": .0001, "clip_norm": 5.0},
        "selection": "highest clean validation ECC; earliest tied epoch; test only after all models train",
        "preprocessing": "matched Resize192, crop160 and flip; ImageNet normalize then J=2 Haar; no clipping or per-image rescaling",
        "initialization": "shared ImageNet1K ResNet18 state and seeded head per seed; dual independent tower storage; W/2 +/- delta classifier blocks, delta~N(0,.01^2), seed900000+seed",
        "test_conditions": {"clean": "center crop"},
        "deterministic_algorithms": True,
    }
    protocol["protocol_hash"] = shared.digest(shared.canonical(protocol))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError(f"Code/data/runtime/settings changed: choose a new output folder: {out}")
    shared.atomic_json(path, protocol)
    for relative in SOURCE_FILES:
        dest = out / "sources" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / relative).read_bytes())
    return datasets, protocol


def verify_completed(result, protocol, initial_hash, plan_hash, dest):
    if (result["protocol_hash"] != protocol["protocol_hash"] or
            result["initial_backbone_sha256"] != initial_hash or
            result["augmentation_plan_sha256"] != plan_hash):
        raise ValueError(f"Completed run identity mismatch: {dest}")
    if file_sha(dest / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {dest}")


def audit_training_complete(out):
    """Fail closed on the exact planned run set before first test evaluation."""
    protocol = read_json(out / "protocol.json")
    args = protocol["args"]
    expected = {f"{method}_seed{seed}" for seed in args["seeds"] for method in args["methods"]}
    actual = {p.parent.name for p in out.glob("*_seed*/result.json")}
    if actual != expected:
        raise ValueError(f"Planned model set incomplete/extra: {out}; missing={expected-actual}, extra={actual-expected}")
    for seed in args["seeds"]:
        initial_hash = shared.state_digest(make_initial(seed, len(protocol["classes"])))
        plan_hash = shared.augmentation_plan(protocol["datasets"]["train"]["n"], args["epochs"], seed)[1]
        for method in args["methods"]:
            dest = out / f"{method}_seed{seed}"
            result, history = read_json(dest / "result.json"), read_json(dest / "history.json")
            verify_completed(result, protocol, initial_hash, plan_hash, dest)
            if result["status"] not in ("trained", "evaluated") or result["method"] != method or result["seed"] != seed:
                raise ValueError(f"Run identity/status invalid: {dest}")
            if [h["epoch"] for h in history] != list(range(1, args["epochs"] + 1)):
                raise ValueError(f"Incomplete training history: {dest}")
            scores = np.array([h["val"]["ecc"] for h in history])
            selected = int(scores.argmax()) + 1
            if not np.isfinite(scores).all() or selected != result["best_epoch"] or scores[selected - 1] != result["best_validation_ecc"]:
                raise ValueError(f"Validation selection invalid: {dest}")
            if not (dest / "val_predictions.npz").is_file():
                raise ValueError(f"Missing selected validation predictions: {dest}")


def fit_one(out, method, seed, state, plans, plan_hash, images, protocol):
    dest = out / f"{method}_seed{seed}"
    initial_hash = shared.state_digest(state)
    if (dest / "result.json").exists():
        result = read_json(dest / "result.json")
        verify_completed(result, protocol, initial_hash, plan_hash, dest)
        print("SKIP_TRAIN", dest, flush=True)
        return result
    dest.mkdir(parents=True, exist_ok=True)
    shared.seed_everything(seed)
    args = protocol["args"]
    model = WaveletEmotionModel(method, state, levels=args["levels"], head_seed=900_000 + seed)
    initial_model_hash = shared.state_digest(model.state_dict())
    classifier_hash = shared.state_digest(model.classifier.state_dict())
    nparams = parameter_count(model)
    model.to(images.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    amp = images.device.type == "cuda"
    scaler = torch.amp.GradScaler(images.device.type, enabled=amp)
    W, polarity = build_cost_matrix(protocol["classes"]), polarity_vector(protocol["classes"])
    best_ecc, best_epoch, best_state, history = -np.inf, None, None, []
    started = time.perf_counter()
    batch_size = args["batch_size"]
    print("START", protocol["dataset"], method, seed, "params", nparams, flush=True)
    for epoch, (order, crop, flip) in enumerate(plans, 1):
        model.train()
        total_loss, steps, skipped, norm_sum = 0., 0, 0, 0.
        for start in range(0, len(order), batch_size):
            stop = min(len(order), start + batch_size)
            x, y = images.batch("train", order[start:stop], crop[start:stop], flip[start:stop])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = nn.functional.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss {dest} epoch{epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient {dest} epoch{epoch}")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            norm_sum += float(norm)
            steps += 1
            total_loss += float(loss.detach()) * len(y)
        vp, vy = shared.predict(model, images, "val", batch_size, amp)
        val = shared.evaluate(vp, vy, W, polarity)
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "val": val,
                        "optimizer_steps": steps - skipped, "amp_skipped_steps": skipped,
                        "mean_gradient_norm_before_clip": norm_sum / steps,
                        "seconds": time.perf_counter() - started})
        if val["ecc"] > best_ecc:
            best_ecc, best_epoch = val["ecc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        shared.atomic_json(dest / "history.json", history)
        print("EPOCH", protocol["dataset"], method, seed, epoch,
              "valECC", round(val["ecc"], 5), "seconds", round(time.perf_counter() - started, 1), flush=True)
    model.load_state_dict(best_state)
    torch.save(best_state, dest / "best.part.pt")
    (dest / "best.part.pt").replace(dest / "best.pt")
    vp, vy = shared.predict(model, images, "val", batch_size, amp)
    shared.save_probabilities(dest / "val_predictions.npz", vp, vy)
    result = {"status": "trained", "method": method, "seed": seed, "protocol_hash": protocol["protocol_hash"],
              "best_epoch": best_epoch, "best_validation_ecc": best_ecc,
              "initial_backbone_sha256": initial_hash, "initial_model_sha256": initial_model_hash,
              "initial_classifier_sha256": classifier_hash, "augmentation_plan_sha256": plan_hash,
              "parameter_count": nparams, "checkpoint_sha256": file_sha(dest / "best.pt"),
              "val": shared.evaluate(vp, vy, W, polarity), "seconds": time.perf_counter() - started,
              "optimizer_steps": sum(h["optimizer_steps"] for h in history),
              "amp_skipped_steps": sum(h["amp_skipped_steps"] for h in history)}
    shared.atomic_json(dest / "result.json", result)
    print("TRAINED", dest, round(result["seconds"], 1), flush=True)
    del model, optimizer, scaler, best_state
    if images.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_and_replay(out, device):
    """Replay validation before measuring the held-out clean test split."""
    protocol = read_json(out / "protocol.json")
    for relative, expected in protocol["sources"].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError(f"Source changed since training: {relative}")
    args = protocol["args"]
    classes = protocol["classes"]
    datasets, metadata = shared.load_dataset(Path(args["data_root"]), classes, shared.SPLITS)
    if metadata != protocol["datasets"]:
        raise ValueError("Dataset changed before evaluation")
    images = shared.Images(datasets, device, True)
    cost, polarity = build_cost_matrix(classes), polarity_vector(classes)
    amp = device.type == "cuda"
    results = []
    for seed in args["seeds"]:
        state = make_initial(seed, len(classes))
        for method in args["methods"]:
            dest = out / f"{method}_seed{seed}"
            record = read_json(dest / "result.json")
            model = WaveletEmotionModel(method, state, levels=args["levels"], head_seed=900_000 + seed).to(device)
            model.load_state_dict(torch.load(dest / "best.pt", map_location=device, weights_only=True))
            val_p, val_y = shared.predict(model, images, "val", args["batch_size"], amp)
            with np.load(dest / "val_predictions.npz", allow_pickle=False) as stored:
                if not np.array_equal(stored["probabilities"], val_p) or not np.array_equal(stored["labels"], val_y):
                    raise ValueError(f"Selected validation predictions did not replay exactly: {dest}")
            test_p, test_y = shared.predict(model, images, "test", args["batch_size"], amp)
            target = dest / "predictions.npz"
            if target.exists():
                with np.load(target, allow_pickle=False) as stored:
                    if not np.array_equal(stored["probabilities"], test_p) or not np.array_equal(stored["labels"], test_y):
                        raise ValueError(f"Previously saved test predictions differ: {target}")
            else:
                shared.save_probabilities(target, test_p, test_y)
            record.update(status="evaluated", validation_replay_exact=True,
                          test=shared.evaluate(test_p, test_y, cost, polarity))
            shared.atomic_json(dest / "result.json", record)
            results.append(record)
    shared.atomic_json(out / "results.json", results)
    return results


def summarize(records):
    scores = {(row["method"], row["seed"]): row["test"]["ecc"] for row in records}
    seeds = sorted(seed for method, seed in scores if method == "rgb_dual")
    if len(scores) != len(records) or set(scores) != {(method, seed) for method in METHODS for seed in seeds}:
        raise ValueError("Expected unique, complete Haar and RGB seed pairs")
    baseline = [scores["rgb_dual", seed] for seed in seeds]
    candidate = [scores["wavelet_fusion", seed] for seed in seeds]
    return {
        "seeds": seeds,
        "rgb_dual_mean_ecc": float(np.mean(baseline)),
        "wavelet_fusion_mean_ecc": float(np.mean(candidate)),
        **paired_seed_summary(baseline, candidate),
    }


def run(args):
    config = load_config(args.config)
    device = configure_runtime(args.device)
    planned = {name: Path(root).resolve() for name, root in config["datasets"].items()}
    with shared.gpu_lock(ROOT / "runs/.experiment.lock"):
        if args.stage in ("all", "train"):
            for name, root in planned.items():
                out = args.out / name
                datasets, protocol = initialize_protocol(name, root, out, config, args.config, device)
                images = shared.Images(datasets, device, True)
                for seed in config["seeds"]:
                    state = make_initial(seed, len(protocol["classes"]))
                    plans, digest = shared.augmentation_plan(len(datasets["train"][1]), config["epochs"], seed)
                    for method in METHODS:
                        fit_one(out, method, seed, state, plans, digest, images, protocol)
                del images, datasets
        if args.stage in ("all", "evaluate"):
            for name in planned:
                out = args.out / name
                protocol = read_json(out / "protocol.json")
                if protocol["config_sha256"] != file_sha(args.config) or protocol["device"] != device.type:
                    raise ValueError(f"Evaluation config or device differs from training: {out}")
                audit_training_complete(out)
            summary = {name: summarize(evaluate_and_replay(args.out / name, device)) for name in planned}
            shared.atomic_json(args.out / "summary.json", summary)
    print(f"Wavelet {args.stage} completed. Artifacts: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wavelet_study.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/wavelet")
    parser.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

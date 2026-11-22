"""Locked, paired multi-resolution experiment and completion-driven GPU pipeline.

This opt-in experiment leaves paper training defaults and earlier studies intact.
All models train before any new test predictions are generated. A killed partial
model is retrained from its recorded initial state; completed models are verified
and skipped. Checkpoints do not claim exact mid-epoch continuation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

import run_seed_evidence as shared
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.wavelets import METHODS, WaveletEmotionModel, wavelet_components, haar_decompose, reconstruct_bands

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/run_wavelet_study.py", "scripts/run_seed_evidence.py",
    "src/ecc/wavelets.py", "src/ecc/emotion.py", "src/ecc/metrics.py", "src/ecc/graph.py",
)
CONDITIONS = ("clean", "shift_x1", "shift_x4", "awgn_003")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    if tuple(config["methods"]) != METHODS:
        raise ValueError("The locked six-method study requires the documented method order")
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


def initialize_protocol(name, data_root, out, config, config_path, benchmark=False):
    classes = (data_root / "classes.txt").read_text(encoding="utf8").splitlines()
    splits = ("train", "val") if benchmark else shared.SPLITS
    datasets, metadata = shared.load_dataset(data_root, classes, splits)
    cache_hashes = {s: file_sha(data_root / f"{s}_seed_evidence_{m['cache_key'][:16]}.npy")
                    for s, m in metadata.items()}
    plan_path = ROOT / "docs/wavelet_study/STUDY_PLAN.md"
    sources = {p: file_sha(ROOT / p) for p in SOURCE_FILES}
    args = {"data_root": str(data_root.relative_to(ROOT)), "epochs": config["epochs"],
            "seeds": config["seeds"], "methods": config["methods"],
            "batch_size": config["batch_size"], "levels": config["levels"]}
    protocol = {
        "dataset": name, "args": args, "classes": classes, "datasets": metadata,
        "cache_sha256": cache_hashes, "sources": sources,
        "data_manifest_sha256": file_sha(data_root / "manifest.json") if (data_root / "manifest.json").exists() else None,
        "config_sha256": file_sha(config_path), "study_plan_path": str(plan_path.relative_to(ROOT)),
        "study_plan_sha256": file_sha(plan_path), "benchmark": benchmark,
        "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "loss": "eight-class cross entropy only, identical for all methods",
        "optimizer": {"name": "AdamW", "lr": .0001, "weight_decay": .0001, "clip_norm": 5.0},
        "selection": "highest clean validation ECC; earliest tied epoch; test only after all models train",
        "preprocessing": "matched Resize192, crop160 and flip; ImageNet normalize then J=2 Haar; no clipping or per-image rescaling",
        "initialization": "shared ImageNet1K ResNet18 state and seeded head per seed; dual independent tower storage; W/2 +/- delta classifier blocks, delta~N(0,.01^2), seed900000+seed",
        "test_conditions": {"clean": "center crop", "shift_x1": "circular horizontal roll by one pixel AFTER crop",
                            "shift_x4": "circular horizontal roll by four pixels AFTER crop",
                            "awgn_003": "unclipped independent N(0,.03^2) raw RGB noise, shared CPU RNG612034 reset for each model/condition; fixed batch order"},
        "fusion_interventions": ["remove_detail", "remove_low"],
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


@torch.no_grad()
def condition_predict(model, images, split, batch_size, condition):
    model.eval()
    probabilities, labels = [], []
    generator = torch.Generator(device="cpu").manual_seed(612034)
    std = torch.tensor([.229, .224, .225], device=images.device).view(1, 3, 1, 1)
    for start in range(0, len(images.labels[split]), batch_size):
        x, y = images.batch(split, np.arange(start, min(len(images.labels[split]), start + batch_size)))
        if condition.startswith("shift_x"):
            x = x.roll(int(condition[7:]), dims=-1)
        elif condition == "awgn_003":
            x = x + .03 * torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device) / std
        elif condition == "remove_detail":
            x = wavelet_components(x, model.levels)[0]
        elif condition == "remove_low":
            x = wavelet_components(x, model.levels)[1]
        elif condition != "clean":
            raise ValueError(condition)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=images.device.type == "cuda"):
            logits = model(x)
        probabilities.append(logits.float().softmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def evaluate_and_replay(out, device, benchmark=False):
    protocol = read_json(out / "protocol.json")
    for relative, expected in protocol["sources"].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError(f"Source changed since training: {relative}")
    datasets, metadata = shared.load_dataset(ROOT / protocol["args"]["data_root"], protocol["classes"], shared.SPLITS)
    if metadata != protocol["datasets"]:
        raise ValueError("Dataset changed before evaluation")
    images = shared.Images(datasets, device, True)
    W, polarity = build_cost_matrix(protocol["classes"]), polarity_vector(protocol["classes"])
    replay, perturbations, results = [], [], []
    for seed in protocol["args"]["seeds"]:
        state = make_initial(seed, len(W))
        for method in protocol["args"]["methods"]:
            dest = out / f"{method}_seed{seed}"
            result = read_json(dest / "result.json")
            verify_completed(result, protocol, shared.state_digest(state),
                             shared.augmentation_plan(len(datasets["train"][1]), protocol["args"]["epochs"], seed)[1], dest)
            model = WaveletEmotionModel(method, state, levels=protocol["args"]["levels"], head_seed=900_000 + seed).to(device)
            model.load_state_dict(torch.load(dest / "best.pt", map_location=device, weights_only=True))
            # Replay independent loading of the selected checkpoint against the
            # validation array saved during training before evaluating test.
            vp, vy = condition_predict(model, images, "val", protocol["args"]["batch_size"], "clean")
            with np.load(dest / "val_predictions.npz", allow_pickle=False) as z:
                maxdiff = float(np.max(np.abs(vp - z["probabilities"])))
                exact = np.array_equal(vp, z["probabilities"]) and np.array_equal(vy, z["labels"])
            if not exact:
                raise ValueError(f"Checkpoint validation replay is not bitwise exact: {dest}, {maxdiff}")
            replay.append({"method": method, "seed": seed, "split": "val", "bitwise_exact": True,
                           "max_probability_difference": maxdiff})
            for condition in CONDITIONS + (("remove_detail", "remove_low") if method == "wavelet_fusion" else ()):
                pred_file = dest / ("predictions.npz" if condition == "clean" else f"predictions_{condition}.npz")
                p, y = condition_predict(model, images, "test", protocol["args"]["batch_size"], condition)
                if pred_file.exists():
                    with np.load(pred_file, allow_pickle=False) as z:
                        if not np.array_equal(z["probabilities"], p) or not np.array_equal(z["labels"], y):
                            raise ValueError(f"Previously saved prediction replay differs: {pred_file}")
                else:
                    shared.save_probabilities(pred_file, p, y)
                values = shared.evaluate(p, y, W, polarity)
                perturbations.append({"method": method, "seed": seed, "condition": condition, "metrics": values,
                                      "file": str(pred_file.relative_to(out)), "sha256": file_sha(pred_file)})
                if condition == "clean":
                    result["test"] = values
            result["status"] = "evaluated"
            shared.atomic_json(dest / "result.json", result)
            results.append(result)
            shared.atomic_json(out / "checkpoint_replay.json", {"status": "passed", "scope": "all completed validation arrays reloaded from best.pt; test generated from same loaded checkpoint", "arrays": replay})
            shared.atomic_json(out / "perturbations.json", perturbations)
            shared.atomic_json(out / "results.json", results)
            print("EVALUATED", protocol["dataset"], method, seed, flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del images, datasets
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def transform_diagnostics(datasets, classes, device):
    """Training inputs only; mathematical identity and grid-sensitivity audits."""
    rng = np.random.default_rng(702914)
    labels = datasets["train"][1]
    selected = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), min(32, int((labels == c).sum())), replace=False)
                                      for c in range(len(classes))]))
    images = shared.Images({"train": datasets["train"]}, device, False)
    class_rows, energies = [], []
    maxima = {"reconstruction_max_absolute": 0., "band_sum_max_absolute": 0.,
              "parseval_relative_error": 0., "low_detail_relative_inner_product": 0.,
              "blockmean_max_absolute": 0., "shift4_commutator_max_absolute": 0.}
    shift1 = []
    for start in range(0, len(selected), 32):
        x, y = images.batch("train", selected[start:start + 32])
        x = x.double()
        low, detail = wavelet_components(x, 2)
        bands = reconstruct_bands(haar_decompose(x, 2))
        total = x.square().flatten(1).sum(1)
        band_energy = {k: v.square().flatten(1).sum(1) for k, v in bands.items()}
        reconstructed = sum(bands.values())
        block = nn.functional.avg_pool2d(x, 4).repeat_interleave(4, -2).repeat_interleave(4, -1)
        low_shift1 = wavelet_components(x.roll(1, -1), 2)[0]
        low_shift4 = wavelet_components(x.roll(4, -1), 2)[0]
        maxima["reconstruction_max_absolute"] = max(maxima["reconstruction_max_absolute"], float((x - low - detail).abs().max()))
        maxima["band_sum_max_absolute"] = max(maxima["band_sum_max_absolute"], float((x - reconstructed).abs().max()))
        maxima["parseval_relative_error"] = max(maxima["parseval_relative_error"], float(((sum(band_energy.values()) - total).abs() / total.clamp_min(1e-30)).max()))
        maxima["low_detail_relative_inner_product"] = max(maxima["low_detail_relative_inner_product"], float(((low * detail).flatten(1).sum(1).abs() / total.clamp_min(1e-30)).max()))
        maxima["blockmean_max_absolute"] = max(maxima["blockmean_max_absolute"], float((low - block).abs().max()))
        maxima["shift4_commutator_max_absolute"] = max(maxima["shift4_commutator_max_absolute"], float((low_shift4 - low.roll(4, -1)).abs().max()))
        centered = x - x.mean((-2, -1), keepdim=True)
        centered_total = centered.square().flatten(1).sum(1)
        centered_low = low - low.mean((-2, -1), keepdim=True)
        for i, index in enumerate(selected[start:start + 32]):
            energies.append({"train_row": int(index), "class": classes[int(y[i])],
                             "normalized_low_energy_fraction": float(low[i].square().sum() / total[i]),
                             "spatially_centered_low_energy_fraction": float(centered_low[i].square().sum() / centered_total[i].clamp_min(1e-30)),
                             **{f"{k}_energy_fraction": float(e[i] / total[i]) for k, e in band_energy.items()},
                             **{f"centered_{k}_energy_fraction": float((centered_low[i].square().sum() if k == "LL2" else e[i]) / centered_total[i].clamp_min(1e-30))
                                for k, e in band_energy.items()}})
        shift1.extend(((low_shift1 - low.roll(1, -1)).square().flatten(1).sum(1) / total.clamp_min(1e-30)).sqrt().cpu().tolist())
    noise = torch.from_numpy(np.random.default_rng(45127).normal(size=(32, 3, 160, 160))).to(device)
    nl, nd = wavelet_components(noise, 2)
    nenergy = float(noise.square().sum())
    for c in classes:
        rows = [r for r in energies if r["class"] == c]
        class_rows.append({"class": c, "n": len(rows), **{key: float(np.mean([r[key] for r in rows]))
            for key in rows[0] if key.endswith("energy_fraction")}})
    del images
    result = {"input_scope": "fixed training-only class-stratified sample; clean center crops; normalized tensor; float64 identity audit",
            "sample_count": len(selected), "selected_train_rows": selected.tolist(), "maxima": maxima,
            "shift1_relative_commutator_mean": float(np.mean(shift1)),
            "noise": {"distribution": "iid N(0,1), float64 fixed numpy seed45127",
                      "expected_low_fraction": 1 / 16, "measured_low_fraction": float(nl.square().sum()) / nenergy,
                      "expected_detail_fraction": 15 / 16, "measured_detail_fraction": float(nd.square().sum()) / nenergy},
            "class_summary": class_rows, "image_energy": energies}
    # These are exact finite-dimensional identities, not performance targets.
    if any(not np.isfinite(value) or value > 1e-10 for value in maxima.values()):
        raise FloatingPointError(f"Haar identity audit failed: {maxima}")
    if abs(result["noise"]["measured_low_fraction"] - 1 / 16) > .002:
        raise FloatingPointError("White-noise projection audit failed")
    if not np.isfinite(result["shift1_relative_commutator_mean"]):
        raise FloatingPointError("Nonfinite shift diagnostic")
    result["status"] = "passed"
    return result


def run(args):
    config = load_config(args.config)
    device = configure_runtime(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    selected_datasets = config["datasets"]
    if args.benchmark:
        selected_datasets = {"fi": selected_datasets["fi"]}
        config = dict(config, epochs=1, seeds=[999], methods=["wavelet_fusion"])
    with shared.gpu_lock(ROOT / "work/single_gpu.lock"):
        protocol_paths = []
        for name, relative in selected_datasets.items():
            datasets, protocol = initialize_protocol(name, ROOT / relative, args.out / name, config, args.config, args.benchmark)
            protocol_paths.append(args.out / name / "protocol.json")
            if args.stage in ("all", "train"):
                if not args.benchmark:
                    diag = transform_diagnostics(datasets, protocol["classes"], device)
                    shared.atomic_json(args.out / name / "transform_diagnostics.json", diag)
                images = shared.Images(datasets, device, True)
                all_results = []
                for seed in config["seeds"]:
                    state = make_initial(seed, len(protocol["classes"]))
                    plans, plan_hash = shared.augmentation_plan(len(datasets["train"][1]), config["epochs"], seed)
                    for method in config["methods"]:
                        all_results.append(fit_one(args.out / name, method, seed, state, plans, plan_hash, images, protocol))
                        shared.atomic_json(args.out / name / "results.json", all_results)
                del images, datasets
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if args.benchmark:
            print("BENCHMARK_COMPLETE validation only", flush=True)
            return
        # Audit every source image against cached inputs before test evaluation.
        if args.stage == "all":
            import verify_seed_inputs
            checks = [verify_seed_inputs.verify_protocol(path, ROOT, require_full_splits=True) for path in protocol_paths]
            shared.atomic_json(args.out / "input_verification.json", {"status": "passed", "datasets": checks})
        if args.stage in ("all", "evaluate"):
            for path in protocol_paths:
                audit_training_complete(path.parent)
            for path in protocol_paths:
                evaluate_and_replay(path.parent, device)
    if args.stage == "all":
        from report_wavelet_study import build_report
        build_report(args.out, bootstrap=1500)
        shared.atomic_json(args.out / "automatic_completion.json", {"status": "complete", "models": len(selected_datasets) * len(config["seeds"]) * len(config["methods"]), "report": str(args.out / "REPORT.html")})
    print("WAVELET_STUDY_COMPLETE", args.stage, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/wavelet_study.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/wavelet_study")
    p.add_argument("--stage", choices=("all", "train", "evaluate"), default="all")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--benchmark", action="store_true", help="Separate one-epoch fusion timing run, validation only; never used for study inference")
    run(p.parse_args())


if __name__ == "__main__":
    main()

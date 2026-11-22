"""Import audit and follow-up frequency study of frozen full-EmoSet checkpoints.

No training or alpha selection is performed on this already-inspected test set.
"""

from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18
from ecc.graph_experiment import Uint8Normalize
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.metrics import evaluate_predictions, graph_error_metrics, confusion_matrix
from ecc.spatial_frequency import transform_images

SOURCE_BYTES = Path(__file__).read_bytes()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf8")


def audit(root, out):
    runs = root / "emoset_graph_alpha_sweep_2gpu/runs"
    data = root / "data/emoset_full"
    classes = (data / "classes.txt").read_text().splitlines()
    W = build_cost_matrix(classes)
    pol = polarity_vector(classes)
    rows = list(csv.DictReader((data / "test.csv").open()))
    truth = np.array([classes.index(r["label"]) for r in rows])
    results = []
    for path in sorted(runs.glob("*/metrics.json")):
        record = json.loads(path.read_text())
        history = json.loads((path.parent / "history.json").read_text())
        assert (
            record["complete"] and record["completed_epochs"] == 8 and len(history) == 8
        )
        assert (
            record["best_epoch"] == int(np.argmax([h["val_ecc"] for h in history])) + 1
        )
        with np.load(path.parent / "predictions.npz") as stored:
            p, y = stored["probs"], stored["labels"]
            assert np.array_equal(y, truth) and np.allclose(p.sum(1), 1, atol=1e-6)
            assert np.isfinite(p).all() and np.all(p >= 0)
            recomputed = evaluate_predictions(y, p.argmax(1), W, pol)
            recomputed.update(graph_error_metrics(p, y, W))
            for key in ["acc", "ecc", "emc", "acc2", "gsee", "brier"]:
                assert np.isclose(recomputed[key], record["test"][key], atol=1e-8), (
                    path,
                    key,
                )
            cm = np.loadtxt(path.parent / "confusion_matrix.csv", delimiter=",")
            assert np.array_equal(cm, confusion_matrix(y, p.argmax(1), 8))
        record["validation_best_ecc"] = max(h["val_ecc"] for h in history)
        results.append(record)
    assert len(results) == 15
    grouped = {}
    for method in sorted({r["method"] for r in results}):
        rs = [r for r in results if r["method"] == method]
        assert sorted(r["seed"] for r in rs) == [0, 1, 2]
        grouped[method] = {
            k: {
                "mean": float(np.mean([r["test"][k] for r in rs])),
                "std": float(np.std([r["test"][k] for r in rs], ddof=1)),
            }
            for k in ["acc", "ecc", "emc", "acc2", "gsee"]
        }
        grouped[method]["validation_mean"] = float(
            np.mean([r["validation_best_ecc"] for r in rs])
        )
    selected = max(
        [m for m in grouped if m != "ce"], key=lambda m: grouped[m]["validation_mean"]
    )
    recorded = json.loads((runs.parent / "validation_alpha_selection.json").read_text())
    assert selected == recorded["selected_graph_method"] == "graph_a0p3"
    report = {
        "verified_runs": 15,
        "epochs_each": 8,
        "test_images_each": len(truth),
        "classes": classes,
        "selected_graph_method": selected,
        "summary": grouped,
        "failure": "All training/evaluation completed; export status print calls stat().st_size() instead of stat().st_size.",
        "notebook_name_warning": "Filename says 25 epoch, but executed MAX_EPOCHS and saved histories are 8.",
    }
    write(out / "kaggle_audit.json", report)
    return classes, rows, report


def run(args):
    torch.set_num_threads(3)
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes, rows, review = audit(root, out)
    rng = np.random.default_rng(20261115)
    all_labels = np.array([classes.index(r["label"]) for r in rows])
    indices = np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(all_labels == c),
                    min(args.per_class, int((all_labels == c).sum()))
                    if args.per_class
                    else int((all_labels == c).sum()),
                    replace=False,
                )
                for c in range(8)
            ]
        )
    )
    y = all_labels[indices]
    cache_digest = hashlib.sha256(
        (root / "data/emoset_full/test.csv").read_bytes() + str(args.per_class).encode()
    ).hexdigest()[:16]
    cache = out / f"input_cache_{cache_digest}.npy"
    selected = [rows[i] for i in indices]
    if not cache.exists():
        temporary_cache = cache.with_suffix(".part.npy")
        a = np.lib.format.open_memmap(
            temporary_cache,
            mode="w+",
            dtype=np.uint8,
            shape=(len(indices), 3, 160, 160),
        )
        for i, row in enumerate(selected):
            with Image.open(root / "data/emoset_full" / row["path"]) as im:
                im = im.convert("RGB")
                assert im.size == (192, 192)
                a[i] = np.asarray(im).transpose(2, 0, 1)[:, 16:176, 16:176]
        a.flush()
        del a
        temporary_cache.replace(cache)
    x = np.load(cache, mmap_mode="c")
    assert x.shape == (len(indices), 3, 160, 160), "Invalid image cache"
    identity = [{"test_index": int(i), **row} for i, row in zip(indices, selected)]
    write(out / "selected_images.json", identity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    replay = {}
    runs = root / "emoset_graph_alpha_sweep_2gpu/runs"
    for method in ["ce", "graph_a0p3"]:
        for seed in [0, 1, 2]:
            name = f"{method}_seed{seed}"
            model = torch.nn.Sequential(
                Uint8Normalize(), resnet18(weights=None, num_classes=8)
            )
            model.load_state_dict(
                torch.load(
                    runs / name / "best.pt", map_location="cpu", weights_only=True
                )
            )
            models[name] = model.to(device).eval()
    W = build_cost_matrix(classes)
    pol = polarity_vector(classes)
    protocol = {
        "dataset": str(root),
        "selection_seed": 20261115,
        "images": len(indices),
        "per_class": args.per_class,
        "sample": "all original test rows; exploratory robustness analysis"
        if args.per_class == 0
        else "balanced random subset of previously inspected test set; exploratory robustness analysis",
        "cutoffs_cycles_per_pixel": [0.025, 0.05, 0.1, 0.2, 0.35, 0.5],
        "butterworth_order": 2,
        "highpass": "1-LPF with DC reset to 1; per-channel mean preserved before clipping",
        "phase_controls": "amplitude preserved before clipping for zero/random phase; flat amplitude keeps original phase and AC energy",
        "boundary": "periodic FFT; possible edge discontinuities",
        "clipping": "clip to [0,255], no min-max rescaling; fraction and mean drift reported",
        "torch": torch.__version__,
        "device": str(device),
        "amp": device.type == "cuda",
        "batch_size": args.batch_size,
        "source_sha256": hashlib.sha256(SOURCE_BYTES).hexdigest(),
        "training": "none; checkpoints frozen",
        "checkpoint_sha256": {
            n: hashlib.sha256((runs / n / "best.pt").read_bytes()).hexdigest()
            for n in models
        },
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Output protocol changed; select a fresh output directory")
    write(protocol_path, protocol)
    (out / "runner_source.py").write_bytes(SOURCE_BYTES)
    import ecc.spatial_frequency as sf

    (out / "spatial_frequency_source.py").write_bytes(Path(sf.__file__).read_bytes())
    conditions = [("clean", None)] + [
        (mode, fc)
        for mode in ["lowpass", "highpass_dc"]
        for fc in protocol["cutoffs_cycles_per_pixel"]
    ]
    conditions += [
        (mode, None) for mode in ["phase_zero", "phase_random", "amplitude_flat"]
    ]
    results = []
    started = time.time()
    for mode, fc in conditions:
        name = mode if fc is None else f"{mode}_{fc:g}"
        dest = out / name
        dest.mkdir(exist_ok=True)
        if (dest / "metrics.json").exists():
            results.extend(json.loads((dest / "metrics.json").read_text()))
            continue
        prediction = {key: [] for key in models}
        mse = []
        clips = []
        drift = []
        generator = torch.Generator(device=device).manual_seed(20261115)
        for start in range(0, len(x), args.batch_size):
            raw = (
                torch.from_numpy(
                    np.array(x[start : start + args.batch_size], copy=True)
                )
                .to(device)
                .float()
            )
            changed = transform_images(raw, mode, fc, generator=generator)
            clips.extend(
                ((changed < 0) | (changed > 255)).float().mean((1, 2, 3)).cpu().tolist()
            )
            changed = changed.clamp(0, 255)
            mse.extend((changed - raw).square().mean((1, 2, 3)).cpu().tolist())
            drift.extend(
                (changed.mean((2, 3)) - raw.mean((2, 3))).abs().mean(1).cpu().tolist()
            )
            for key, model in models.items():
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ),
                ):
                    p = model(changed).float().softmax(1).cpu().numpy()
                prediction[key].append(p)
        condition_results = []
        for key, chunks in prediction.items():
            p = np.concatenate(chunks)
            m = evaluate_predictions(y, p.argmax(1), W, pol)
            m.update(graph_error_metrics(p, y, W))
            cm = confusion_matrix(y, p.argmax(1), 8)
            m["balanced_acc"] = float(np.mean(np.diag(cm) / cm.sum(1)) * 100)
            polarity_correct = pol[y] == pol[p.argmax(1)]
            m["balanced_acc2"] = float(
                50
                * (
                    polarity_correct[pol[y] == 0].mean()
                    + polarity_correct[pol[y] == 1].mean()
                )
            )
            if mode == "clean":
                original = np.load(runs / key / "predictions.npz")["probs"][indices]
                replay[key] = {
                    "max_probability_difference": float(abs(p - original).max()),
                    "argmax_agreement": float(
                        (p.argmax(1) == original.argmax(1)).mean()
                    ),
                }
            np.savez_compressed(dest / f"{key}.npz", probabilities=p, labels=y)
            method, seed = key.rsplit("_seed", 1)
            record = {
                "condition": name,
                "mode": mode,
                "cutoff": fc,
                "method": method,
                "seed": int(seed),
                "test": m,
                "mean_mse": float(np.mean(mse)),
                "mean_clipped_fraction": float(np.mean(clips)),
                "mean_channel_drift": float(np.mean(drift)),
            }
            condition_results.append(record)
        write(dest / "metrics.json", condition_results)
        results.extend(condition_results)
        print(
            name,
            "elapsed",
            round(time.time() - started, 1),
            "ECC",
            [
                (r["method"], r["seed"], round(r["test"]["ecc"], 3))
                for r in condition_results
            ],
            flush=True,
        )
    write(out / "results.json", results)
    write(out / "checkpoint_replay.json", replay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/kaggle_full_8epoch")
    ap.add_argument("--out", default="outputs/kaggle_recovery_fft")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    run(ap.parse_args())


if __name__ == "__main__":
    main()

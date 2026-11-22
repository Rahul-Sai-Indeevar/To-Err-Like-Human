"""Independently verify and summarize the opt-in paired wavelet study (CPU only).

No model selection occurs here. All contrasts, metrics and multiplicity families
are fixed before the new outcomes; training seeds remain the replication units.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.seed_statistics import paired_seed_summary, crossed_seed_image_bootstrap

_helper_spec = importlib.util.spec_from_file_location(
    "wavelet_report_seed_helpers", Path(__file__).with_name("report_seed_evidence.py"))
_helper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_helper)
verify_predictions = _helper.verify_predictions
holm = _helper.holm
plan_digest = _helper.plan_digest
sample_summary = _helper.sample_summary

METHODS = ("rgb", "low", "detail", "wavelet_fusion", "rgb_dual", "low_dual")
METRICS = ("acc", "ecc", "acc2", "emc", "brier", "gsee")
PRIMARY = (("wavelet_fusion", "rgb_dual"),)
SECONDARY = (("wavelet_fusion", "rgb"), ("wavelet_fusion", "low_dual"),
             ("low", "rgb"), ("detail", "rgb"))
LABELS = {"rgb": "RGB single", "low": "Approximation only", "detail": "Details only",
          "wavelet_fusion": "Approximation + details", "rgb_dual": "RGB + RGB control",
          "low_dual": "Approximation + approximation control"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _require_hash(path, expected, description):
    if not path.is_file() or file_digest(path) != expected:
        raise ValueError(f"{description} hash mismatch: {path}")


def _safe_relative(root, name):
    value = (root / name).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes repository: {name}")
    return value


def _expected_labels(data_root, split, classes, metadata):
    path = data_root / f"{split}.csv"
    _require_hash(path, metadata["csv_sha256"], f"{split} CSV")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([classes.index(row["label"].lower()) for row in rows], dtype=np.int64)
    if len(labels) != metadata["n"] or np.bincount(labels, minlength=len(classes)).tolist() != metadata["class_counts"]:
        raise ValueError(f"{split}: manifest label counts mismatch")
    expected_key = hashlib.sha256(path.read_bytes() + b"PIL_EXIF_RGB_torchvisionResize192_antialias_v1").hexdigest()
    if metadata.get("cache_key") != expected_key:
        raise ValueError(f"{split}: cache key mismatch")
    return labels


def _metric_contrasts(runs, predictions, labels, seeds, W, bootstrap):
    output = {}
    for candidate, baseline in (*PRIMARY, *SECONDARY):
        name = f"{candidate}_minus_{baseline}"
        metrics = {metric: paired_seed_summary(
            [runs[(baseline, seed)]["test"][metric] for seed in seeds],
            [runs[(candidate, seed)]["test"][metric] for seed in seeds]) for metric in METRICS}
        contributions = np.stack([
            100 / W[labels, predictions[(candidate, seed)].argmax(1)] -
            100 / W[labels, predictions[(baseline, seed)].argmax(1)] for seed in seeds])
        output[name] = {"candidate": candidate, "baseline": baseline,
                        "family": "primary" if (candidate, baseline) in PRIMARY else "secondary",
                        "metrics": metrics,
                        "ecc_crossed_bootstrap": crossed_seed_image_bootstrap(
                            contributions, labels, repeats=bootstrap, seed=20261118)}
    return output


def analyze_dataset(directory, repo_root, bootstrap=1500):
    """Validate every artifact used in a dataset's clean-test comparisons."""
    directory, repo_root = Path(directory), Path(repo_root).resolve()
    protocol = read_json(directory / "protocol.json")
    unsigned = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != protocol.get("protocol_hash"):
        raise ValueError("Protocol hash mismatch")
    args = protocol["args"]
    methods, seeds = args["methods"], args["seeds"]
    if args["levels"] != 2 or args["epochs"] < 1 or args["batch_size"] < 2:
        raise ValueError("Report requires the locked J=2 partition and positive training budget")
    if len(methods) != len(set(methods)) or set(methods) != set(METHODS):
        raise ValueError("All six uniquely named planned methods are required")
    if not 2 <= len(seeds) <= 20 or len(seeds) != len(set(seeds)):
        raise ValueError("Between two and twenty unique paired seeds are required for exact sign-flip inference")
    if set(protocol["datasets"]) != {"train", "val", "test"}:
        raise ValueError("Report requires all train/validation/test splits")
    data_root = (repo_root / args["data_root"]).resolve()
    classes = protocol["classes"]
    if (data_root / "classes.txt").read_text(encoding="utf-8-sig").splitlines() != classes:
        raise ValueError("Class order mismatch")
    _require_hash(data_root / "manifest.json", protocol["data_manifest_sha256"], "Dataset preparation manifest")
    _require_hash(_safe_relative(repo_root, protocol.get("study_plan_path", "docs/wavelet_study/STUDY_PLAN.md")),
                  protocol["study_plan_sha256"], "Study plan")
    if not protocol["sources"]:
        raise ValueError("Missing executed source hashes")
    for relative, expected in protocol["sources"].items():
        _require_hash(_safe_relative(repo_root, relative), expected, "Current source")
        snapshot = _safe_relative(directory / "sources", relative)
        _require_hash(snapshot, expected, "Executed source snapshot")
    labels = {split: _expected_labels(data_root, split, classes, meta)
              for split, meta in protocol["datasets"].items()}
    for split, expected in protocol["cache_sha256"].items():
        key = protocol["datasets"][split]["cache_key"]
        path = data_root / f"{split}_seed_evidence_{key[:16]}.npy"
        _require_hash(path, expected, f"{split} input cache")
        cache = np.load(path, mmap_mode="r", allow_pickle=False)
        if cache.shape != (len(labels[split]), 3, 192, 192) or cache.dtype != np.uint8:
            raise ValueError(f"{split}: input cache shape/dtype mismatch")
        del cache
    if set(protocol["cache_sha256"]) != set(labels):
        raise ValueError("Every split must have a pinned input cache hash")
    W, polarity = build_cost_matrix(classes), polarity_vector(classes)
    runs, predictions, rows, class_rows = {}, {}, [], []
    checkpoint_count, total_optimizer_steps, total_amp_skips = 0, 0, 0
    minibatches_per_epoch = (len(labels["train"]) + args["batch_size"] - 1) // args["batch_size"]
    for seed in seeds:
        backbone_hashes, plan_hashes = set(), set()
        group_states = {"single": set(), "dual": set()}
        group_classifiers = {"single": set(), "dual": set()}
        group_parameters = {"single": set(), "dual": set()}
        for method in methods:
            path = directory / f"{method}_seed{seed}"
            result = read_json(path / "result.json")
            if result.get("status") != "evaluated" or "test" not in result:
                raise ValueError(f"Unfinished test evaluation: {path}")
            if result["method"] != method or result["seed"] != seed or result["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError(f"Run identity/protocol mismatch: {path}")
            backbone_hashes.add(result["initial_backbone_sha256"])
            plan_hashes.add(result["augmentation_plan_sha256"])
            if result["augmentation_plan_sha256"] != plan_digest(len(labels["train"]), args["epochs"], seed):
                raise ValueError(f"Augmentation plan mismatch: {path}")
            group = "single" if method in ("rgb", "low", "detail") else "dual"
            group_states[group].add(result["initial_model_sha256"])
            group_classifiers[group].add(result["initial_classifier_sha256"])
            group_parameters[group].add(result["parameter_count"])
            history = read_json(path / "history.json")
            if [h["epoch"] for h in history] != list(range(1, args["epochs"] + 1)):
                raise ValueError(f"Incomplete epoch history: {path}")
            if not all(np.isfinite(h["train_loss"]) for h in history):
                raise ValueError(f"Nonfinite training loss: {path}")
            for h in history:
                if (h["optimizer_steps"] < 0 or h["amp_skipped_steps"] < 0 or
                        h["optimizer_steps"] + h["amp_skipped_steps"] != minibatches_per_epoch):
                    raise ValueError(f"Optimizer step accounting mismatch: {path}")
            if (result["optimizer_steps"] != sum(h["optimizer_steps"] for h in history) or
                    result["amp_skipped_steps"] != sum(h["amp_skipped_steps"] for h in history)):
                raise ValueError(f"Optimizer result/history mismatch: {path}")
            total_optimizer_steps += result["optimizer_steps"]
            total_amp_skips += result["amp_skipped_steps"]
            validation_ecc = np.array([h["val"]["ecc"] for h in history])
            if not np.isfinite(validation_ecc).all():
                raise ValueError(f"Nonfinite validation ECC: {path}")
            best = int(np.argmax(validation_ecc))
            if result["best_epoch"] != best + 1 or not np.isclose(result["best_validation_ecc"], validation_ecc[best], atol=1e-7):
                raise ValueError(f"Validation checkpoint selection mismatch: {path}")
            _, _, val = verify_predictions(path / "val_predictions.npz", result["val"], labels["val"], W, polarity)
            if not np.isclose(val["ecc"], result["best_validation_ecc"], atol=1e-6):
                raise ValueError(f"Selected checkpoint validation ECC mismatch: {path}")
            p, y, metrics = verify_predictions(path / "predictions.npz", result["test"], labels["test"], W, polarity)
            if not result.get("checkpoint_sha256"):
                raise ValueError(f"Missing checkpoint hash: {path}")
            if (path / "best.pt").exists():
                _require_hash(path / "best.pt", result["checkpoint_sha256"], "Checkpoint")
                checkpoint_count += 1
            predictions[(method, seed)], runs[(method, seed)] = p, result
            class_ecc = []
            for c, name in enumerate(classes):
                mask = y == c
                if not mask.any():
                    raise ValueError("Each test class must be represented for macro diagnostics")
                values = {"dataset": directory.name, "method": method, "seed": seed, "class": name,
                          "count": int(mask.sum()), "acc": float(100 * (p.argmax(1)[mask] == c).mean()),
                          "ecc": float((100 / W[y[mask], p.argmax(1)[mask]]).mean())}
                class_rows.append(values)
                class_ecc.append(values["ecc"])
            rows.append({"dataset": directory.name, "method": method, "seed": seed,
                         "best_epoch": result["best_epoch"], "parameter_count": result["parameter_count"],
                         "seconds": result["seconds"], "macro_ecc": float(np.mean(class_ecc)), **{m: metrics[m] for m in METRICS}})
        if len(backbone_hashes) != 1 or len(plan_hashes) != 1:
            raise ValueError(f"Paired backbone initial state/augmentation mismatch for seed {seed}")
        if any(len(values) != 1 for group in (group_states, group_classifiers, group_parameters) for values in group.values()):
            raise ValueError(f"Parameter-matched initial model/classifier mismatch for seed {seed}")
        if next(iter(group_parameters["dual"])) <= next(iter(group_parameters["single"])):
            raise ValueError("Dual control must contain two independently trainable backbones")
    method_summary = {method: {metric: sample_summary([r[metric] for r in rows if r["method"] == method])
                               for metric in (*METRICS, "macro_ecc", "seconds", "best_epoch")}
                      for method in methods}
    summary = {"dataset": directory.name, "seeds": seeds, "n_seed_pairs": len(seeds),
               "split_counts": {s: len(y) for s, y in labels.items()}, "classes": classes,
               "method_summary": method_summary, "comparisons": _metric_contrasts(runs, predictions, labels["test"], seeds, W, bootstrap),
               "parameters": {m: runs[(m, seeds[0])]["parameter_count"] for m in methods},
               "training": {"epochs": args["epochs"], "batch_size": args["batch_size"], "levels": args["levels"],
                            "optimizer": protocol.get("optimizer"), "loss": protocol.get("loss"),
                            "selection": protocol.get("selection")},
               "data_provenance": _helper.dataset_provenance(data_root, protocol),
               "protocol_hash": protocol["protocol_hash"],
               "seed_scope": "Paired independently trained seeds on one fixed data split; no independent training-data population replication"}
    verification = {"status": "passed", "protocol_hash": protocol["protocol_hash"],
                    "verified_runs": len(runs), "prediction_arrays_verified": 2 * len(runs),
                    "checkpoints_hashed": checkpoint_count, "checkpoint_replay_performed_by_this_report": False,
                    "source_snapshots_verified": len(protocol["sources"]), "input_caches_hashed": len(labels),
                    "optimizer_steps": total_optimizer_steps, "amp_skipped_steps": total_amp_skips,
                    "planned_minibatches_per_run": minibatches_per_epoch * args["epochs"],
                    "validation_selection": "Highest validation ECC, earliest exact tie, full planned epoch history",
                    "paired_controls": "Shared backbone initialization and augmentation plan; identical initial model/classifier and parameter counts within single/dual groups",
                    "source_image_pixel_audit_performed_by_this_report": False}
    return summary, verification, rows, class_rows


def adjust_families(datasets):
    """ECC-only Holm families: one capacity contrast/dataset; four secondary/dataset."""
    output = {}
    for family in ("primary", "secondary"):
        entries = [(name, key, c) for name, d in datasets.items() for key, c in d["comparisons"].items() if c["family"] == family]
        adjusted = holm([c["metrics"]["ecc"]["exact_signflip"]["p_two_sided"] for _, _, c in entries])
        output[family] = []
        for (name, key, comp), p in zip(entries, adjusted):
            comp["ecc_holm_adjusted_p"] = p
            output[family].append({"dataset": name, "comparison": key, "adjusted_p": p})
    return output


def conclusion(comp):
    stats = comp["metrics"]["ecc"]
    interval = stats["t95"]
    direction = "increase" if stats["mean_difference"] > 0 else "decrease" if stats["mean_difference"] < 0 else "difference of zero"
    if interval is None:
        uncertainty = "The seed interval is unavailable."
    elif interval[0] <= 0 <= interval[1]:
        uncertainty = "The seed t95 interval includes zero; the direction is uncertain."
    else:
        uncertainty = "The seed t95 interval excludes zero under its normality assumptions."
    adjusted = comp.get("ecc_holm_adjusted_p", 1.0)
    decision = "rejects" if adjusted < .05 else "does not reject"
    return (f"The observed {direction} is {abs(stats['mean_difference']):.3f} ECC points. "
            f"{uncertainty} The ECC sign-flip comparison {decision} the within-seed exchangeability null "
            f"after its predeclared Holm correction (p={adjusted:.4g}). "
            "A non-rejection does not establish equality or a zero effect; fixed-split seed uncertainty does not include new training datasets.")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(out, datasets):
    figure, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.4), squeeze=False)
    for axis, (name, dataset) in zip(axes[0], datasets.items()):
        positions = np.arange(len(METHODS))
        means = [dataset["method_summary"][m]["ecc"]["mean"] for m in METHODS]
        sd = [dataset["method_summary"][m]["ecc"]["sample_sd"] or 0 for m in METHODS]
        axis.bar(positions, means, yerr=sd, capsize=3, color=["#72869e", "#7ea788", "#d1ad69", "#336a98", "#a6a3bb", "#a9bfac"])
        axis.set_xticks(positions, ["RGB", "Low", "Detail", "Fusion", "RGB dual", "Low dual"], rotation=40, ha="right")
        axis.set_title(f"{name}: mean ± seed SD")
        axis.set_ylabel("Clean-test ECC (%)")
        axis.set_ylim(0, 100)
    figure.tight_layout()
    figure.savefig(out / "method_ecc.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.4, max(3, len(datasets) * .8 + 1.8)))
    for i, (name, dataset) in enumerate(datasets.items()):
        stats = dataset["comparisons"]["wavelet_fusion_minus_rgb_dual"]["metrics"]["ecc"]
        mean, interval = stats["mean_difference"], stats["t95"]
        if interval is not None:
            axis.errorbar(mean, i, xerr=np.array([[mean - interval[0]], [interval[1] - mean]]), fmt="o", capsize=5, color="#285f8d")
        else:
            axis.plot(mean, i, "o", color="#285f8d")
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(datasets)), list(datasets))
    axis.set_xlabel("Fusion − equally sized RGB dual: ECC points (paired seed t95)")
    axis.set_title("Primary capacity-controlled comparison; fixed dataset splits")
    axis.grid(axis="x", alpha=.2)
    figure.tight_layout()
    figure.savefig(out / "primary_effects.png", dpi=170)
    plt.close(figure)


def _markdown(summary):
    lines = ["# Wavelet multiresolution emotion-recognition experiment", "",
             "This is an opt-in experiment on fixed real-image splits. It evaluates a spatial signal-processing inductive bias; it does not establish a human visual mechanism or reproduce the complete NeurIPS benchmark.", "",
             "## What is being tested", "",
             "A two-level orthonormal Haar filter bank separates each normalized 160×160 RGB image into its same-size LL2 reconstruction and all signed details. LL2 reconstruction equals a replicated 4×4 block mean. Analysis and synthesis reconstruct the original input, and squared subband energies sum to input energy (Parseval). The low and detail projections are orthogonal and sum to the original image. Fusion therefore adds no information; any benefit must come from how the model uses the representation.", "",
             "All six methods use cross-entropy. Approximation-only and details-only models identify which components support recognition. The main comparison is approximation+detail fusion against a parameter-matched RGB+RGB network. Approximation+approximation controls for two-branch capacity without complementary details. Dual classifier blocks use matched antisymmetric initialization perturbations to avoid identical RGB towers remaining identical during training.", "",
             "No per-image min-max scaling or clipping is applied to signed details. Decomposition acts on ImageNet-normalized input z: low=Pz and detail=(I−P)z; this avoids adding an artificial RGB mean to the detail branch. Both towers see full-size reconstructed components, so this design does not claim the compute savings of a coefficient-domain network.", "",
             "A decimated Haar projection is shift-variant: a one-pixel shift changes the block alignment. Retaining only LL2 is not an ideal low-pass filter, and this architecture does not remove aliasing from later CNN downsampling. The experiment is distinct from WaveCNet, WTConv and scattering networks.", "",
             "## Data and training budget", "",
             "All models in a dataset use the same fixed split, paired pretrained backbone/head initialization and precomputed crop/flip/minibatch plans. Checkpoints use only validation ECC. No inspected test result changes the training settings or the planned comparisons.", "",
             "| Dataset | Train / validation / test images | Epochs | Batch | Paired seeds |", "|---|---|---:|---:|---|"]
    for name, dataset in summary["datasets"].items():
        counts, training = dataset["split_counts"], dataset["training"]
        lines.append(f"| {name} | {counts['train']:,} / {counts['val']:,} / {counts['test']:,} | {training['epochs']} | {training['batch_size']} | {', '.join(map(str, dataset['seeds']))} |")
    lines += ["", "Dataset preparation records, source/archive fingerprints, class counts and split provenance are preserved in `dataset_provenance.json`. These prepared subsets and fixed splits must not be presented as the canonical full FI, ArtPhoto or EmoSet benchmark protocols.", "",
             "## Clean-test results", "",
             "Numbers below are averages across paired independently trained seeds; ± denotes the sample standard deviation across seeds. ACC2 measures polarity accuracy. EMC is conditional on the model's changing subset of mistakes and is not a physical signal-to-noise ratio.", "",
             "| Dataset | Method | Seeds | ACC | ECC | ACC2 | EMC | Brier | Parameters |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        for method in METHODS:
            metrics = dataset["method_summary"][method]
            cells = [f"{metrics[m]['mean']:.3f} ± {metrics[m]['sample_sd']:.3f}" for m in ("acc", "ecc", "acc2", "emc")]
            lines.append(f"| {name} | {LABELS[method]} | {dataset['n_seed_pairs']} | " + " | ".join(cells) + f" | {metrics['brier']['mean']:.5f} | {dataset['parameters'][method]:,} |")
    lines += ["", "![Clean-test ECC](method_ecc.png)", "", "## Primary comparison", "",
              "The predeclared primary contrast is fusion minus RGB dual ECC, with one contrast per dataset in a single Holm family. Positive values favor fusion. The Student interval uses paired seed differences; the crossed bootstrap additionally resamples shared class-stratified test images, preserving the pairing across methods and seeds.", "",
              "| Dataset | ECC difference | Paired seed t95 | Crossed seed/image 95% | Wins | Exact sign-flip p | Holm p |", "|---|---:|---|---|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        comp = dataset["comparisons"]["wavelet_fusion_minus_rgb_dual"]
        stats, bootstrap = comp["metrics"]["ecc"], comp["ecc_crossed_bootstrap"]
        interval = stats["t95"]
        seed_ci = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "unavailable"
        crossed = bootstrap["ci95"]
        lines.append(f"| {name} | {stats['mean_difference']:+.3f} | {seed_ci} | [{crossed[0]:+.3f}, {crossed[1]:+.3f}] | {stats['positive_pairs']}/{stats['n_seed_pairs']} | {stats['exact_signflip']['p_two_sided']:.4g} | {comp['ecc_holm_adjusted_p']:.4g} |")
    lines += ["", "![Primary paired effects](primary_effects.png)", ""]
    for name, dataset in summary["datasets"].items():
        lines += [f"**{name}:** {conclusion(dataset['comparisons']['wavelet_fusion_minus_rgb_dual'])}", ""]
    lines += ["## Secondary contrasts", "",
              "Four predeclared ECC contrasts per dataset form a separate Holm family. They explain the representation rather than replacing an unsuccessful primary comparison. ACC, ACC2, EMC, Brier, GSEE and macro-ECC are descriptive secondary metrics with no additional success claims.", "",
              "| Dataset | Contrast | ECC difference | Paired seed t95 | Holm p |", "|---|---|---:|---|---:|"]
    for name, dataset in summary["datasets"].items():
        for candidate, baseline in SECONDARY:
            comp = dataset["comparisons"][f"{candidate}_minus_{baseline}"]
            stats, interval = comp["metrics"]["ecc"], comp["metrics"]["ecc"]["t95"]
            ci = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else "unavailable"
            lines.append(f"| {name} | {candidate} − {baseline} | {stats['mean_difference']:+.3f} | {ci} | {comp['ecc_holm_adjusted_p']:.4g} |")
    lines += ["", "## Verification and limits", "",
              "The report independently checks protocol, study-plan and executed-source hashes; split manifests and cached input arrays; shared initialization and augmentation; full epoch histories; earliest validation-ECC checkpoint selection; and every saved validation/test probability array with metric recomputation. Available checkpoint files are hash-checked. This report itself does not replay model inference or compare source-image pixels; separate audits are identified explicitly when present.", "",
              f"Verified {summary['total_models']} trained models and {summary['total_prediction_arrays']} clean probability arrays. Dataset splits are analyzed separately; their training seeds and test images are not pooled into a fictitious larger replication count.", "",
              "The same existing test splits have been inspected in earlier experiments, so this is a transparently planned extension rather than a pristine unseen-benchmark confirmation. Training-seed uncertainty is conditional on the chosen data split, pretraining, optimizer and epoch budget. A small seed study does not establish a population-wide effect. Non-significance is not equivalence, and a positive ECC difference does not prove psychological plausibility.", "",
              "Haar multiresolution theory motivates the partition and measurable energy/shift predictions; it does not guarantee emotion-recognition gains. Gaussian-noise and shift/branch-ablation diagnostics, if produced separately, are descriptive checks and cannot be used to retune the reported models.", "",
              "## Reproduction artifacts", "",
              "Use the study runner for sequential single-GPU training and its evaluation/replay stages. Regenerate this report with `python scripts/report_wavelet_study.py --out outputs/wavelet_study --bootstrap 1500`. The package contains saved probabilities, histories, protocol/source fingerprints and tables; raw photographs, caches and model weights remain local.", "",
              "Primary theory: [Mallat, A Theory for Multiresolution Signal Decomposition (1989)](https://www.di.ens.fr/~mallat/papiers/MallatTheory89.pdf). Related architectures, not reproduced here: [WaveCNet](https://arxiv.org/abs/2107.13335), [WTConv](https://arxiv.org/abs/2407.05848).", ""]
    if summary["diagnostics_artifacts"]:
        lines += ["Separate audit/diagnostic artifacts found and fingerprinted:", ""]
        for name, entry in summary["diagnostics_artifacts"].items():
            lines.append(f"- `{name}` (SHA256 `{entry['sha256']}`); inspect its own status and scope before claiming its checks passed.")
    return "\n".join(lines) + "\n"


def build_report(out, bootstrap=1500, repo_root=None):
    out = Path(out).resolve()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    paths = sorted(out.glob("*/protocol.json"))
    if not paths:
        raise ValueError("No dataset protocols found under output directory")
    datasets, verification, rows, class_rows = {}, {}, [], []
    for path in paths:
        dataset, audit, per_seed, per_class = analyze_dataset(path.parent, repo_root, bootstrap)
        datasets[path.parent.name], verification[path.parent.name] = dataset, audit
        rows.extend(per_seed)
        class_rows.extend(per_class)
    families = adjust_families(datasets)
    artifacts = {name: {"sha256": file_digest(out / name)} for name in
                 ("input_verification.json", "transform_diagnostics.json", "checkpoint_replay.json", "perturbations.json") if (out / name).exists()}
    summary = {"datasets": datasets, "multiplicity_families": families,
               "total_models": sum(v["verified_runs"] for v in verification.values()),
               "total_prediction_arrays": sum(v["prediction_arrays_verified"] for v in verification.values()),
               "bootstrap_repeats": bootstrap, "diagnostics_artifacts": artifacts,
               "primary_endpoint": "ECC: wavelet_fusion minus parameter-matched rgb_dual",
               "uncertainty_unit": "Paired independently trained seed on a fixed split; crossed bootstrap additionally resamples shared images"}
    write_json(out / "summary.json", summary)
    write_json(out / "verification.json", verification)
    write_json(out / "dataset_provenance.json", {name: d["data_provenance"] for name, d in datasets.items()})
    write_csv(out / "seed_results.csv", rows)
    write_csv(out / "per_class_seed_metrics.csv", class_rows)
    compare_rows = []
    for name, dataset in datasets.items():
        for key, comp in dataset["comparisons"].items():
            stats, interval = comp["metrics"]["ecc"], comp["metrics"]["ecc"]["t95"]
            compare_rows.append({"dataset": name, "comparison": key, "family": comp["family"],
                                 "n_seed_pairs": stats["n_seed_pairs"], "ecc_difference": stats["mean_difference"],
                                 "seed_t95_low": interval[0] if interval else None, "seed_t95_high": interval[1] if interval else None,
                                 "crossed95_low": comp["ecc_crossed_bootstrap"]["ci95"][0], "crossed95_high": comp["ecc_crossed_bootstrap"]["ci95"][1],
                                 "positive_pairs": stats["positive_pairs"], "raw_signflip_p": stats["exact_signflip"]["p_two_sided"],
                                 "holm_p": comp["ecc_holm_adjusted_p"]})
    write_csv(out / "summary.csv", compare_rows)
    _plots(out, datasets)
    markdown = _markdown(summary)
    (out / "REPORT.md").write_text(markdown, encoding="utf8")
    try:
        import markdown as markdown_library
        body = markdown_library.markdown(markdown, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + html.escape(markdown) + '</pre><img src="method_ecc.png"><img src="primary_effects.png">'
    for name in ("method_ecc.png", "primary_effects.png"):
        encoded = base64.b64encode((out / name).read_bytes()).decode()
        body = body.replace(f'src="{name}"', f'src="data:image/png;base64,{encoded}"')
    document = ("<!doctype html><html><head><meta charset='utf-8'><title>Wavelet study</title>"
                "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 25px;color:#243443}"
                "h1,h2{line-height:1.25}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}"
                "td,th{border:1px solid #d6dce0;padding:7px 10px;text-align:left}th{background:#eef3f6}"
                "img{max-width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#195888}</style></head><body>" + body + "</body></html>")
    (out / "REPORT.html").write_text(document, encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/wavelet_study"))
    parser.add_argument("--bootstrap", type=int, default=1500)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("At least 100 bootstrap draws are required")
    summary = build_report(args.out, args.bootstrap)
    print(f"Verified {summary['total_models']} models; report: {args.out / 'REPORT.html'}")


if __name__ == "__main__":
    main()

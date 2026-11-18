"""CPU verification of fixed wavelet theory, replay and sensitivity diagnostics.

The descriptive condition changes are paired within each frozen trained model.
This supplement performs no training, model selection or additional significance
tests, and does not execute another GPU checkpoint replay.
"""
from __future__ import annotations

import argparse
import base64
import html
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_spec = importlib.util.spec_from_file_location(
    "wavelet_diagnostics_report_helpers", Path(__file__).with_name("report_wavelet_study.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
read_json, write_json = _base.read_json, _base.write_json
file_digest, canonical = _base.file_digest, _base.canonical
METHODS, LABELS = _base.METHODS, _base.LABELS
CONDITIONS = ("clean", "shift_x1", "shift_x4", "awgn_003")
INTERVENTIONS = ("remove_detail", "remove_low")
METRICS = ("ecc", "acc", "acc2")
BANDS = ("LL2", "L1_LH", "L1_HL", "L1_HH", "L2_LH", "L2_HL", "L2_HH")
IDENTITIES = ("reconstruction_max_absolute", "band_sum_max_absolute", "parseval_relative_error",
              "low_detail_relative_inner_product", "blockmean_max_absolute", "shift4_commutator_max_absolute")
CONDITION_LABELS = {"clean": "Clean", "shift_x1": "Circular shift +1 px", "shift_x4": "Circular shift +4 px",
                    "awgn_003": "Unclipped Gaussian noise σ=.03", "remove_detail": "Fusion: detail zeroed",
                    "remove_low": "Fusion: low zeroed"}


def _number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {name}")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"Out-of-range {name}: {value}")
    return float(value)


def verify_transform_diagnostics(path, train_labels, classes):
    """Check exact-identity bounds, grid sample identity and energy accounting."""
    diagnostic = read_json(path)
    if diagnostic.get("status") != "passed":
        raise ValueError("Transform diagnostics do not report a passed status")
    if set(diagnostic["maxima"]) != set(IDENTITIES):
        raise ValueError("Incomplete Haar identity audit")
    for key, value in diagnostic["maxima"].items():
        if _number(value, key, lower=0) >= 1e-10:
            raise ValueError(f"Haar identity exceeds tolerance: {key}")
    _number(diagnostic["shift1_relative_commutator_mean"], "one-pixel commutator", 0, 2)
    noise = diagnostic["noise"]
    low = _number(noise["measured_low_fraction"], "white-noise low fraction", 0, 1)
    detail = _number(noise["measured_detail_fraction"], "white-noise detail fraction", 0, 1)
    if (not np.isclose(noise["expected_low_fraction"], 1 / 16, rtol=0, atol=1e-15) or
            not np.isclose(noise["expected_detail_fraction"], 15 / 16, rtol=0, atol=1e-15)):
        raise ValueError("White-noise expected fractions must be 1/16 and 15/16")
    if abs(low + detail - 1) > 1e-10 or abs(low - 1 / 16) > .002:
        raise ValueError("White-noise projection energy accounting mismatch")
    rng = np.random.default_rng(702914)
    selected = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(train_labels == c), min(32, int((train_labels == c).sum())), replace=False)
        for c in range(len(classes))])).tolist()
    if (diagnostic["selected_train_rows"] != selected or diagnostic["sample_count"] != len(selected) or
            len(diagnostic["image_energy"]) != len(selected)):
        raise ValueError("Training-only diagnostic sample identity/count mismatch")
    expected_keys = {"normalized_low_energy_fraction", "spatially_centered_low_energy_fraction"}
    expected_keys |= {f"{band}_energy_fraction" for band in BANDS}
    expected_keys |= {f"centered_{band}_energy_fraction" for band in BANDS}
    for index, row in zip(selected, diagnostic["image_energy"]):
        if row["train_row"] != index or row["class"] != classes[int(train_labels[index])]:
            raise ValueError("Training energy row label/order mismatch")
        if {k for k in row if k.endswith("energy_fraction")} != expected_keys:
            raise ValueError("Missing or unexpected subband energy fractions")
        for key in expected_keys:
            _number(row[key], key, -1e-12, 1 + 1e-10)
        raw_sum = sum(row[f"{band}_energy_fraction"] for band in BANDS)
        centered_sum = sum(row[f"centered_{band}_energy_fraction"] for band in BANDS)
        # A spatially constant image has zero centered energy; the runner's
        # clamped denominator then records all centered fractions as zero.
        if (abs(raw_sum - 1) > 1e-10 or min(abs(centered_sum), abs(centered_sum - 1)) > 1e-10 or
                abs(row["normalized_low_energy_fraction"] - row["LL2_energy_fraction"]) > 1e-10 or
                abs(row["spatially_centered_low_energy_fraction"] - row["centered_LL2_energy_fraction"]) > 1e-10):
            raise ValueError("Per-image subband energy partition mismatch")
        raw_details = np.array([row[f"{band}_energy_fraction"] for band in BANDS[1:]])
        centered_details = np.array([row[f"centered_{band}_energy_fraction"] for band in BANDS[1:]])
        # Removing a spatial constant changes only the LL2 coefficient. All six
        # detail-band energies therefore change by one common denominator ratio.
        if (row["spatially_centered_low_energy_fraction"] > row["normalized_low_energy_fraction"] + 1e-10 or
                (centered_sum < 1e-10 and raw_details.sum() > 1e-10)):
            raise ValueError("Raw/centered energy consistency mismatch")
        if centered_details.sum() > 1e-10:
            denominator_ratio = raw_details.sum() / centered_details.sum()
            if denominator_ratio > 1 + 1e-10 or not np.allclose(raw_details, denominator_ratio * centered_details, rtol=0, atol=1e-10):
                raise ValueError("Raw/centered detail-band energy consistency mismatch")
        elif raw_details.sum() > 1e-10:
            raise ValueError("Raw/centered detail-band energy consistency mismatch")
    summaries = diagnostic["class_summary"]
    if len(summaries) != len(classes) or {r["class"] for r in summaries} != set(classes):
        raise ValueError("Class energy summary coverage mismatch")
    for summary in summaries:
        matching = [r for r in diagnostic["image_energy"] if r["class"] == summary["class"]]
        if summary["n"] != len(matching) or not matching:
            raise ValueError("Class energy sample count mismatch")
        for key in expected_keys:
            _number(summary[key], f"class mean {key}", -1e-12, 1 + 1e-10)
            if not np.isclose(summary[key], np.mean([r[key] for r in matching]), rtol=0, atol=1e-10):
                raise ValueError("Class energy summary does not match per-image values")
    return diagnostic


def verify_replay_records(path, methods, seeds):
    replay = read_json(path)
    if replay.get("status") != "passed":
        raise ValueError("Checkpoint replay does not report a passed status")
    expected = {(method, seed, "val") for method in methods for seed in seeds}
    found = set()
    for row in replay["arrays"]:
        identity = (row["method"], row["seed"], row["split"])
        if identity not in expected or identity in found:
            raise ValueError("Duplicate or unexpected checkpoint replay record")
        if row.get("bitwise_exact") is not True or _number(row["max_probability_difference"], "replay difference", 0) != 0:
            raise ValueError("Checkpoint validation replay is not recorded as bitwise exact")
        found.add(identity)
    if found != expected:
        raise ValueError("Incomplete checkpoint replay coverage")
    return replay


def _expected_conditions(methods, seeds):
    return {(method, seed, condition) for method in methods for seed in seeds
            for condition in CONDITIONS + (INTERVENTIONS if method == "wavelet_fusion" else ())}


def analyze_diagnostics_dataset(directory, repo_root):
    """Verify all condition arrays and summarize within-model metric changes."""
    import hashlib
    directory, repo_root = Path(directory), Path(repo_root).resolve()
    protocol = read_json(directory / "protocol.json")
    unsigned = {k: v for k, v in protocol.items() if k != "protocol_hash"}
    if hashlib.sha256(canonical(unsigned)).hexdigest() != protocol.get("protocol_hash"):
        raise ValueError("Diagnostic protocol hash mismatch")
    methods, seeds = protocol["args"]["methods"], protocol["args"]["seeds"]
    if len(methods) != len(set(methods)) or set(methods) != set(METHODS) or len(seeds) != len(set(seeds)) or len(seeds) < 2:
        raise ValueError("Diagnostics require six unique methods and unique paired seeds")
    if protocol["args"]["levels"] != 2:
        raise ValueError("Diagnostic theory is locked to J=2 Haar")
    classes = protocol["classes"]
    data_root = (repo_root / protocol["args"]["data_root"]).resolve()
    if (data_root / "classes.txt").read_text(encoding="utf-8-sig").splitlines() != classes:
        raise ValueError("Diagnostic class order mismatch")
    _base._require_hash(data_root / "manifest.json", protocol["data_manifest_sha256"], "Dataset preparation manifest")
    _base._require_hash(_base._safe_relative(repo_root, protocol["study_plan_path"]), protocol["study_plan_sha256"], "Study plan")
    for relative, expected_hash in protocol["sources"].items():
        _base._require_hash(_base._safe_relative(repo_root, relative), expected_hash, "Current source")
        _base._require_hash(_base._safe_relative(directory / "sources", relative), expected_hash, "Executed source snapshot")
    labels = {split: _base._expected_labels(data_root, split, classes, meta)
              for split, meta in protocol["datasets"].items()}
    if set(labels) != {"train", "val", "test"}:
        raise ValueError("Diagnostic protocol requires all data splits")
    transform = verify_transform_diagnostics(directory / "transform_diagnostics.json", labels["train"], classes)
    replay = verify_replay_records(directory / "checkpoint_replay.json", methods, seeds)
    W, polarity = _base.build_cost_matrix(classes), _base.polarity_vector(classes)
    runs, validation_hashes = {}, {}
    for method in methods:
        for seed in seeds:
            path = directory / f"{method}_seed{seed}"
            result = read_json(path / "result.json")
            if (result.get("status") != "evaluated" or result["method"] != method or result["seed"] != seed or
                    result["protocol_hash"] != protocol["protocol_hash"]):
                raise ValueError("Diagnostic run identity/status/protocol mismatch")
            _base.verify_predictions(path / "val_predictions.npz", result["val"], labels["val"], W, polarity)
            validation_hashes[f"{method}_seed{seed}"] = file_digest(path / "val_predictions.npz")
            runs[(method, seed)] = result
    records = read_json(directory / "perturbations.json")
    expected, values, hashes = _expected_conditions(methods, seeds), {}, {}
    for record in records:
        identity = (record["method"], record["seed"], record["condition"])
        if identity not in expected or identity in values:
            raise ValueError("Duplicate or unexpected perturbation record")
        method, seed, condition = identity
        filename = "predictions.npz" if condition == "clean" else f"predictions_{condition}.npz"
        expected_path = (directory / f"{method}_seed{seed}" / filename).resolve()
        actual_path = _base._safe_relative(directory, record["file"])
        if actual_path != expected_path:
            raise ValueError("Perturbation record points to a different run or condition")
        _base._require_hash(actual_path, record["sha256"], "Perturbation probability array")
        _, _, metrics = _base.verify_predictions(actual_path, record["metrics"], labels["test"], W, polarity)
        if condition == "clean":
            _base.verify_predictions(actual_path, runs[(method, seed)]["test"], labels["test"], W, polarity)
        values[identity] = metrics
        hashes[str(actual_path.relative_to(directory))] = record["sha256"]
    if set(values) != expected:
        raise ValueError("Incomplete perturbation condition coverage")
    rows, condition_summary = [], {}
    for method in methods:
        condition_summary[method] = {}
        for condition in CONDITIONS + (INTERVENTIONS if method == "wavelet_fusion" else ()):
            changes = {metric: [values[(method, seed, condition)][metric] - values[(method, seed, "clean")][metric]
                                for seed in seeds] for metric in METRICS}
            condition_summary[method][condition] = {
                "n_seed_pairs": len(seeds), "seeds": seeds,
                "metrics": {metric: _base.sample_summary([values[(method, seed, condition)][metric] for seed in seeds]) for metric in METRICS},
                "change_from_clean": {metric: {**_base.sample_summary(delta), "per_seed": delta,
                                                 "negative_pairs": int(np.count_nonzero(np.array(delta) < 0)),
                                                 "positive_pairs": int(np.count_nonzero(np.array(delta) > 0))}
                                      for metric, delta in changes.items()}}
            for seed in seeds:
                rows.append({"dataset": directory.name, "method": method, "seed": seed, "condition": condition,
                             **{metric: values[(method, seed, condition)][metric] for metric in _base.METRICS},
                             **{f"{metric}_change_from_clean": values[(method, seed, condition)][metric] - values[(method, seed, "clean")][metric] for metric in METRICS}})
    summary = {"dataset": directory.name, "seeds": seeds, "n_seed_pairs": len(seeds),
               "condition_summary": condition_summary,
               "transform": {key: value for key, value in transform.items() if key != "image_energy"},
               "protocol_hash": protocol["protocol_hash"],
               "metric_change_unit": "Condition minus clean in percentage points, paired within the same frozen model",
               "replay_scope": replay.get("scope"),
               "limitations": "Descriptive fixed probes on previously inspected fixed test splits; no extra significance tests or model selection; no independent new training-data replication"}
    verification = {"status": "passed", "protocol_hash": protocol["protocol_hash"],
                    "transform_identity_maxima": transform["maxima"], "training_diagnostic_images": transform["sample_count"],
                    "validation_replay_records_verified": len(replay["arrays"]),
                    "saved_validation_arrays_recomputed": len(validation_hashes), "validation_array_sha256": validation_hashes,
                    "test_condition_arrays_sha_and_metrics_verified": len(hashes), "condition_array_sha256": hashes,
                    "gpu_replay_executed_by_this_report": False,
                    "scope": "Checks recorded exact validation replay coverage and independently recomputes saved-array metrics; test predictions were generated from the loaded checkpoints, not independently replayed by this CPU report",
                    "artifact_sha256": {name: file_digest(directory / name) for name in
                                         ("protocol.json", "transform_diagnostics.json", "checkpoint_replay.json", "perturbations.json")}}
    return summary, verification, rows


def _plots(out, datasets):
    figure, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), squeeze=False)
    colors = {"shift_x1": "#2d6b98", "shift_x4": "#7d63a7", "awgn_003": "#b77b32"}
    for axis, (name, dataset) in zip(axes[0], datasets.items()):
        for j, condition in enumerate(CONDITIONS[1:]):
            positions = np.arange(len(METHODS)) + (j - 1) * .22
            mean = [dataset["condition_summary"][m][condition]["change_from_clean"]["ecc"]["mean"] for m in METHODS]
            sd = [dataset["condition_summary"][m][condition]["change_from_clean"]["ecc"]["sample_sd"] for m in METHODS]
            axis.errorbar(positions, mean, yerr=sd, fmt="o", capsize=3, color=colors[condition], label=CONDITION_LABELS[condition])
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(range(len(METHODS)), ["RGB", "Low", "Detail", "Fusion", "RGB dual", "Low dual"], rotation=40, ha="right")
        axis.set_ylabel("Condition − clean ECC (percentage points)")
        axis.set_title(f"{name}: paired mean ± seed SD")
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "wavelet_sensitivity.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(len(datasets), 2, figsize=(13, 4 * len(datasets)), squeeze=False)
    for i, (name, dataset) in enumerate(datasets.items()):
        classes = dataset["transform"]["class_summary"]
        for j, field in enumerate(("normalized_low_energy_fraction", "spatially_centered_low_energy_fraction")):
            axis = axes[i, j]
            low = [row[field] for row in classes]
            band_prefix = "" if j == 0 else "centered_"
            detail = [sum(row[f"{band_prefix}{band}_energy_fraction"] for band in BANDS[1:]) for row in classes]
            axis.bar(range(len(classes)), low, color="#618caa", label="LL2 approximation")
            axis.bar(range(len(classes)), detail, bottom=low, color="#d0b36f", label="All six detail subbands")
            axis.set_xticks(range(len(classes)), [r["class"] for r in classes], rotation=35, ha="right")
            axis.set_ylim(0, 1.02)
            axis.set_ylabel("Mean per-image squared-energy fraction")
            axis.set_title(f"{name}: " + ("ImageNet-normalized, uncentered" if j == 0 else "Spatial means removed per channel"))
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "wavelet_train_energy.png", dpi=170)
    plt.close(figure)


def _markdown(summary):
    lines = ["# Wavelet theory and frozen-model sensitivity diagnostics", "",
             "This supplement connects exact signal-processing predictions to measured behavior. The prediction probes are fixed descriptive diagnostics; they do not select checkpoints, retune models or introduce additional significance claims.", "",
             "## What the theory predicts", "",
             "For two-level orthonormal Haar analysis, the reconstructed LL2 component is an orthogonal projection P onto 4×4 block constants. The signed complement is (I−P)x. They sum to x, are orthogonal, and their subband squared energies obey Parseval. LL2 reconstruction is a replicated block average, not an ideal frequency cutoff. These identities are verified on a fixed training-only sample with float64 arithmetic.", "",
             "The LL2 subspace has 1/16 of the dimensions. Therefore isotropic iid zero-mean white noise has expected low-component energy fraction 1/16 and detail fraction 15/16. This is a projection-energy prediction on synthetic noise, not a claim about class information, classification accuracy or physical image SNR.", "",
             "A one-pixel circular horizontal shift generally changes the decimation grid. A four-pixel circular shift commutes with P on this divisible periodic grid. Commutation of preprocessing does not imply that a strided CNN's predictions are invariant to the shift.", "",
             "| Dataset | Training images audited | Largest identity residual | One-pixel commutator mean | White-noise low energy | White-noise detail energy |", "|---|---:|---:|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        transform = dataset["transform"]
        lines.append(f"| {name} | {transform['sample_count']} | {max(transform['maxima'].values()):.3g} | {transform['shift1_relative_commutator_mean']:.5f} | {transform['noise']['measured_low_fraction']:.6f} | {transform['noise']['measured_detail_fraction']:.6f} |")
    lines += ["", "The largest residual mixes absolute reconstruction/block-mean residuals with dimensionless relative energy/inner-product residuals; each individual value and its units are retained in `diagnostics_summary.json`. The four-pixel commutator is one of the separately checked near-zero identity residuals.", "",
              "## Where natural-image energy lies", "",
              "The first panel uses the ImageNet-normalized tensor without spatial centering. The second removes each channel's spatial mean before forming energy fractions. Both are averages of per-image fractions, then summarized by class on training inputs only. The uncentered low energy includes DC/color offsets; energy concentration is not evidence that the same fraction of emotion information lies there.", "",
              "![Training-only energy partitions](wavelet_train_energy.png)", "",
              "| Dataset | Class | Images | Uncentered normalized LL2 energy | Spatially centered LL2 energy |", "|---|---|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        for row in dataset["transform"]["class_summary"]:
            lines.append(f"| {name} | {row['class']} | {row['n']} | {row['normalized_low_energy_fraction']:.5f} | {row['spatially_centered_low_energy_fraction']:.5f} |")
    lines += ["", "## Fixed shift and noise probes", "",
              "All probes use the same validation-selected frozen model. Shifts are circular horizontal rolls after the center crop. Gaussian noise is additive iid N(0,.03²) in RGB pixel units where the nominal range is 0…1, with the CPU random generator reset to the same seed and batch order for every model. Noise is not clipped; values can leave the nominal pixel range. These probes do not represent all real-world corruptions.", "",
              "Values are condition minus clean in percentage points. A negative value is a drop. ± denotes the sample standard deviation across the within-model paired seed changes, not a confidence interval or independent image-level replication.", "",
              "![Frozen-model sensitivity](wavelet_sensitivity.png)", "",
              "| Dataset | Method | Condition | ECC change | ACC change | ACC2 change |", "|---|---|---|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        for method in METHODS:
            for condition in CONDITIONS[1:]:
                change = dataset["condition_summary"][method][condition]["change_from_clean"]
                cells = [f"{change[metric]['mean']:+.3f} ± {change[metric]['sample_sd']:.3f}" for metric in METRICS]
                lines.append(f"| {name} | {LABELS[method]} | {CONDITION_LABELS[condition]} | " + " | ".join(cells) + " |")
    lines += ["", "## Fusion component sensitivity", "",
              "Removing detail projects the normalized tensor onto P before the fusion model, so its detail branch receives zero. Removing low uses (I−P), so its approximation branch receives zero up to arithmetic roundoff. Zero here is in normalized component coordinates; it does not mean setting a raw RGB image to black. These are out-of-distribution, nonlinear branch sensitivity probes. Drops cannot be interpreted as a causal information attribution, and the two drops need not add to the clean score.", "",
              "| Dataset | Intervention | ECC change | ACC change | ACC2 change |", "|---|---|---:|---:|---:|"]
    for name, dataset in summary["datasets"].items():
        for condition in INTERVENTIONS:
            change = dataset["condition_summary"]["wavelet_fusion"][condition]["change_from_clean"]
            cells = [f"{change[metric]['mean']:+.3f} ± {change[metric]['sample_sd']:.3f}" for metric in METRICS]
            lines.append(f"| {name} | {CONDITION_LABELS[condition]} | " + " | ".join(cells) + " |")
    lines += ["", "## Verification scope", "",
              f"Verified the declared hashes and independently recomputed metrics for {summary['test_condition_arrays_verified']} test-condition arrays. Also checked complete bitwise-exact validation replay records for {summary['validation_replay_records_verified']} models and independently recomputed their saved validation-array metrics. These arrays include the clean arrays already checked in the main report; overlapping counts must not be added as independent evidence.", "",
              "The CPU supplement checks that every planned method/seed/condition appears exactly once, every replay record reports zero probability difference, class/split identities agree with the pinned CSVs, and clean condition metrics agree with each run's final test record. It also checks protocol/executed-source fingerprints, training sample indices, per-image subband energy partitions, class means, noise-energy accounting and identity residual bounds.", "",
              "The actual checkpoint reloading and validation inference replay occurred in the GPU evaluation stage. This CPU report validates that recorded audit and the saved predictions; it does not independently rerun network inference. Test conditions were generated from those loaded selected checkpoints. No test replay beyond the explicitly recorded validation replay is claimed.", "",
              "Results remain conditional on the existing fixed splits, ImageNet initialization, optimizer and short training budget. Training seeds are not independent dataset samples, and ArtPhoto is small. Detail-only results can be affected by ImageNet input adaptation and source JPEG/resizing artifacts. Haar LL2 is exactly a block-average projection, so the experiment does not establish that Haar uniquely outperforms other filters or discover a human visual mechanism.", "",
              "Machine-readable outputs: `diagnostics_summary.json`, `diagnostics_verification.json`, `condition_results.csv`. Regenerate with `python scripts/report_wavelet_diagnostics.py --out outputs/wavelet_study` after the evaluation stage completes.", ""]
    return "\n".join(lines) + "\n"


def build_diagnostics(out, repo_root=None):
    out = Path(out).resolve()
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    protocols = sorted(out.glob("*/protocol.json"))
    if not protocols:
        raise ValueError("No dataset protocols found")
    datasets, verification, rows = {}, {}, []
    for path in protocols:
        dataset, audit, condition_rows = analyze_diagnostics_dataset(path.parent, repo_root)
        datasets[path.parent.name], verification[path.parent.name] = dataset, audit
        rows.extend(condition_rows)
    summary = {"datasets": datasets,
               "test_condition_arrays_verified": sum(v["test_condition_arrays_sha_and_metrics_verified"] for v in verification.values()),
               "validation_replay_records_verified": sum(v["validation_replay_records_verified"] for v in verification.values()),
               "inference_scope": "Descriptive paired seed changes, no new significance tests or pooled dataset inference"}
    write_json(out / "diagnostics_summary.json", summary)
    write_json(out / "diagnostics_verification.json", verification)
    _base.write_csv(out / "condition_results.csv", rows)
    _plots(out, datasets)
    markdown = _markdown(summary)
    (out / "DIAGNOSTICS.md").write_text(markdown, encoding="utf8")
    try:
        import markdown as markdown_library
        body = markdown_library.markdown(markdown, extensions=["tables", "fenced_code"])
    except ImportError:
        body = '<pre>' + html.escape(markdown) + '</pre><img src="wavelet_train_energy.png"><img src="wavelet_sensitivity.png">'
    for name in ("wavelet_train_energy.png", "wavelet_sensitivity.png"):
        encoded = base64.b64encode((out / name).read_bytes()).decode()
        body = body.replace(f'src="{name}"', f'src="data:image/png;base64,{encoded}"')
    document = ("<!doctype html><html><head><meta charset='utf-8'><title>Wavelet diagnostics</title>"
                "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 25px;color:#243443}"
                "table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{border:1px solid #d6dce0;padding:7px 10px}"
                "th{background:#eef3f6}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style></head><body>" + body + "</body></html>")
    (out / "DIAGNOSTICS.html").write_text(document, encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/wavelet_study"))
    args = parser.parse_args()
    summary = build_diagnostics(args.out)
    print(f"Verified {summary['test_condition_arrays_verified']} test-condition arrays; supplement: {args.out / 'DIAGNOSTICS.html'}")


if __name__ == "__main__":
    main()

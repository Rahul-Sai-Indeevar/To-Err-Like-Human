"""Verify fresh paired-seed studies and write separate dataset-level evidence.

Run from the repository: python scripts/report_seed_evidence.py --out outputs/seed_evidence
No GPU computation occurs in this report. Existing studies remain separate.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.metrics import evaluate_predictions, graph_error_metrics
from ecc.seed_statistics import paired_seed_summary, crossed_seed_image_bootstrap
from ecc.graph import graph_basis, spectral_weights

METRICS = ("acc", "ecc", "acc2", "emc", "brier", "gsee")
METHODS = ("ce", "graph", "isotropic", "permuted")
COMPARISONS = (("graph", "ce"), ("graph", "isotropic"), ("graph", "permuted"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def holm(pvalues):
    """Holm familywise adjusted p-values, retaining input order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    previous = 0.0
    for i, j in enumerate(order):
        previous = max(previous, min(1.0, (len(p) - i) * p[j]))
        adjusted[j] = previous
    return adjusted.tolist()


def verify_predictions(path, saved, expected_labels, W, polarity):
    """Independent saved-array checks and metric recomputation."""
    with np.load(path, allow_pickle=False) as z:
        p = z["probabilities"].copy()
        y = z["labels"].copy()
    if p.shape != (len(expected_labels), len(W)) or y.shape != expected_labels.shape:
        raise ValueError(f"Incorrect shapes: {path}")
    if not np.array_equal(y, expected_labels):
        raise ValueError(f"Label identities/order disagree with CSV: {path}")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=2e-6):
        raise ValueError(f"Invalid probabilities: {path}")
    values = evaluate_predictions(y, p.argmax(1), W, polarity)
    values.update(graph_error_metrics(p, y, W))
    for key in (*METRICS, "dirichlet"):
        if not np.isclose(values[key], saved[key], rtol=1e-7, atol=1e-6):
            raise ValueError(f"Metric mismatch {key}: {path}: {values[key]} vs {saved[key]}")
    stored_energy = np.asarray(saved["error_energy"])
    if stored_energy.shape != (len(W),) or not np.allclose(values["error_energy"], stored_energy, rtol=1e-7, atol=1e-9):
        raise ValueError(f"Metric mismatch error_energy: {path}")
    return p, y, values


def plan_digest(n, epochs, seed):
    """Independently reconstruct the runner's fixed PCG64 augmentation plan."""
    rng = np.random.default_rng(400_000 + seed)
    digest = hashlib.sha256()
    for _ in range(epochs):
        parts = (rng.permutation(n).astype(np.int64),
                 rng.integers(0, 33, size=(n, 2), dtype=np.int64), rng.random(n) < .5)
        for part in parts:
            digest.update(part.tobytes())
    return digest.hexdigest()


def sample_summary(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else None}


def eigenspace_groups(values):
    """Group repeated positive eigenvalues; total group energy is invariant."""
    values = np.asarray(values)
    groups = []
    for i in np.flatnonzero(values > 1e-10):
        if groups and np.isclose(values[i], values[groups[-1][0]], rtol=1e-8, atol=1e-10):
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def secondary_diagnostics(predictions, labels, runs, seeds, classes, W, methods=METHODS, comparisons=COMPARISONS):
    """Descriptive class and correct-graph spectral diagnostics; no extra tests."""
    _, _, eigenvalues, _ = graph_basis(W)
    groups = eigenspace_groups(eigenvalues)
    mode_summary, class_rows, macro_by_seed = {}, [], {}
    y = np.asarray(labels)
    for method in methods:
        mode_summary[method] = []
        for indices in groups:
            energies = [float(np.array(runs[(method, seed)]["test"]["error_energy"])[indices].sum()) for seed in seeds]
            mode_summary[method].append({"indices": indices, "multiplicity": len(indices),
                                         "eigenvalue": float(eigenvalues[indices].mean()), **sample_summary(energies)})
        macro_by_seed[method] = {}
        for seed in seeds:
            p = predictions[(method, seed)]
            prediction = p.argmax(1)
            class_ecc = []
            for c, class_name in enumerate(classes):
                mask = y == c
                if not mask.any():
                    raise ValueError("Per-class diagnostics require every declared class in the test split")
                row = {"method": method, "seed": seed, "class": class_name, "count": int(mask.sum()),
                       "acc": float(100 * (prediction[mask] == c).mean()),
                       "ecc": float(np.mean(100 / W[y[mask], prediction[mask]]))}
                class_rows.append(row)
                class_ecc.append(row["ecc"])
            macro_by_seed[method][seed] = float(np.mean(class_ecc))
            energy = np.asarray(runs[(method, seed)]["test"]["error_energy"])
            if not np.isclose(energy.sum(), runs[(method, seed)]["test"]["brier"], atol=1e-8):
                raise ValueError("Parseval check failed for saved error energy")
            if not np.isclose(energy @ spectral_weights(eigenvalues), runs[(method, seed)]["test"]["gsee"], atol=1e-8):
                raise ValueError("GSEE mode-weighting check failed")
    per_class_summary = []
    indexed = {(row["method"], row["seed"], row["class"]): row for row in class_rows}
    delta_rows, delta_summary = [], []
    for method in methods:
        for class_name in classes:
            matching = [indexed[(method, seed, class_name)] for seed in seeds]
            per_class_summary.append({"method": method, "class": class_name, "count": matching[0]["count"],
                                      "acc": sample_summary([r["acc"] for r in matching]),
                                      "ecc": sample_summary([r["ecc"] for r in matching])})
    for method, baseline in comparisons:
        for class_name in classes:
            matching = []
            for seed in seeds:
                a, b = indexed[(baseline, seed, class_name)], indexed[(method, seed, class_name)]
                row = {"comparison": f"{method}_minus_{baseline}", "seed": seed, "class": class_name,
                       "count": a["count"], "acc": b["acc"] - a["acc"], "ecc": b["ecc"] - a["ecc"]}
                delta_rows.append(row); matching.append(row)
            delta_summary.append({"comparison": f"{method}_minus_{baseline}", "class": class_name,
                                  "count": matching[0]["count"], "acc": sample_summary([r["acc"] for r in matching]),
                                  "ecc": sample_summary([r["ecc"] for r in matching])})
    roughness = {method: {metric: sample_summary([runs[(method, seed)]["test"][metric] for seed in seeds])
                         for metric in ("dirichlet", "brier", "gsee")} for method in methods}
    return {"evaluation_graph": "Same correct emotion graph for all four methods, including the wrong-label training control",
            "eigenvalues": eigenvalues.tolist(), "non_dc_eigenspace_energy": mode_summary,
            "energy_note": "Repeated eigenvalues are summed within each eigenspace; DC is omitted. Error energy and GSEE are related to the optimized objective, not independent validation.",
            "probability_roughness": roughness,
            "per_class_seed_metrics": class_rows, "per_class_summary": per_class_summary,
            "per_class_seed_differences": delta_rows, "per_class_difference_summary": delta_summary,
            "balanced_macro_ecc": {method: {**sample_summary(list(values.values())), "per_seed": values}
                                   for method, values in macro_by_seed.items()},
            "macro_note": "Diagnostic arithmetic mean of class-wise ECC; distinct from primary image-weighted ECC; no new inferential tests or class selection"}


def dataset_provenance(data_root, protocol):
    """Pin the complete preparation record while tolerating historical schemas."""
    path = data_root / "manifest.json"
    expected_hash = protocol.get("data_manifest_sha256")
    if expected_hash is None:
        raise ValueError("Study protocol must pin data_manifest_sha256")
    contents = path.read_bytes()
    actual_hash = sha256(contents)
    if actual_hash != expected_hash:
        raise ValueError(f"Dataset preparation manifest hash mismatch: {path}")
    manifest = json.loads(contents.decode("utf-8-sig"))
    source = manifest.get("source", {})
    raw_counts = manifest.get("raw_counts")
    raw_rows = sum(raw_counts.values()) if isinstance(raw_counts, dict) else None
    raw_scope = "raw archive image rows"
    if raw_rows is None and all(isinstance(manifest.get(s), dict) and "available_rows" in manifest[s] for s in ("train", "val", "test")):
        raw_rows = sum(manifest[s]["available_rows"] for s in ("train", "val", "test"))
        raw_scope = "rows available in selected mirror shards, not the complete original dataset"
    retained = manifest.get("retained_counts")
    retained_rows = sum(retained.values()) if isinstance(retained, dict) else None
    mirror = manifest.get("mirror", manifest.get("source_repository", source.get("repository")))
    revision = manifest.get("revision", source.get("revision"))
    source_url = manifest.get("source_url")
    if source_url is None and mirror and revision:
        source_url = f"https://huggingface.co/datasets/{mirror}/tree/{revision}"
    preparation = manifest.get("image_transform", manifest.get("processing"))
    preparation_note = "preparation manifest"
    if preparation is None and mirror == "udeman/EmoSet-118K":
        preparation = "EXIF-transposed RGB; thumbnail at most320x320 LANCZOS; JPEG quality95"
        preparation_note = "project scripts/prepare_graph_data.py; not recorded in the older manifest schema"
    if preparation is None:
        preparation = "Not specified in preparation manifest"
    shards = source.get("files", [])
    if not shards and isinstance(manifest.get("source_inventory"), dict):
        shards = [{"path": filename, **entry} for filename, entry in manifest["source_inventory"].items()]
    if retained_rows is None and "source_shards" in manifest and all("images" in manifest.get(s, {}) for s in ("train", "val", "test")):
        retained_rows = sum(manifest[s]["images"] for s in ("train", "val", "test"))
    if raw_rows is None:
        raw_scope = "raw source row count not recorded in preparation manifest"
    return {"manifest_path": str(path.resolve()), "manifest_sha256": actual_hash,
            "protocol_data_manifest_sha256": expected_hash, "manifest_hash_verified": True,
            "manifest_verbatim": manifest,
            "summary": {"raw_rows": raw_rows, "raw_scope": raw_scope,
                        "usable_before_caps": retained_rows,
                        "mirror": mirror, "pinned_revision": revision, "source_url": source_url,
                        "archive": manifest.get("archive"), "archive_sha256": manifest.get("archive_sha256"),
                        "shards": shards, "prepared_image_transform": preparation,
                        "preparation_description_source": preparation_note,
                        "classifier_input_pipeline": protocol.get("transforms", "Not recorded in protocol"),
                        "split_rule": manifest.get("split_rule", "Preserve full mirror source splits after encoded-byte deduplication" if "source_shards" in manifest else "Preserved mirror splits with per-class shard-subset caps; not the complete benchmark"),
                        "exclusion_limits": manifest.get("exclusion_scope", "Corrupt images and exact encoded-byte duplicates rejected; re-encoded, pixel-identical and near duplicates remain possible" if "source_shards" in manifest else "Exact decoded-pixel duplicate checking during capped subset preparation; near/re-encoded duplicates and uninspected images remain possible"),
                        "actual_split_counts": {s: protocol["datasets"][s]["n"] for s in ("train", "val", "test")},
                        "actual_class_counts": {s: protocol["datasets"][s]["class_counts"] for s in ("train", "val", "test")}}}


def analyze_dataset(directory, repo, repeats, meaningful, methods=METHODS, comparisons=COMPARISONS):
    protocol = read_json(directory / "protocol.json")
    canonical = dict(protocol)
    protocol_hash = canonical.pop("protocol_hash")
    if sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()) != protocol_hash:
        raise ValueError(f"Protocol hash mismatch: {directory}")
    snapshot_hashes = {}
    for key, filename in (("runner", "runner_source.py"), ("ecc.graph", "graph_source.py"),
                          ("ecc.metrics", "metrics_source.py"), ("ecc.emotion", "emotion_source.py")):
        source = directory / filename
        if sha256(source.read_bytes()) != protocol["sources"][key]:
            raise ValueError(f"Executed source snapshot mismatch: {source}")
        snapshot_hashes[key] = protocol["sources"][key]
    classes = protocol["classes"]
    W, polarity = build_cost_matrix(classes), polarity_vector(classes)
    data_root = Path(protocol["args"]["data_root"])
    if not data_root.is_absolute():
        data_root = repo / data_root
    provenance = dataset_provenance(data_root, protocol)
    expected = {}
    for split in ("train", "val", "test"):
        csv_path = data_root / f"{split}.csv"
        if sha256(csv_path.read_bytes()) != protocol["datasets"][split]["csv_sha256"]:
            raise ValueError(f"Dataset CSV changed: {csv_path}")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected[split] = np.array([classes.index(row["label"].lower()) for row in rows])
        if len(rows) != protocol["datasets"][split]["n"]:
            raise ValueError("Dataset count mismatch")
        if np.bincount(expected[split], minlength=len(classes)).tolist() != protocol["datasets"][split]["class_counts"]:
            raise ValueError("Dataset class counts mismatch")
    seeds = sorted(protocol["args"]["seeds"])
    expected_methods = tuple(protocol["args"]["methods"])
    if set(expected_methods) != set(methods):
        raise ValueError(f"A complete study with methods {tuple(methods)} is required for the final report")
    if any(method not in methods or baseline not in methods for method, baseline in comparisons):
        raise ValueError("Comparison uses a method outside the requested study")
    runs, predictions, histories = {}, {}, {}
    max_epochs = protocol["args"]["epochs"]
    for seed in seeds:
        hashes = []
        plan = plan_digest(len(expected["train"]), max_epochs, seed)
        for method in methods:
            run_dir = directory / f"{method}_seed{seed}"
            if not run_dir.exists():
                run_dir = directory / "runs" / f"{method}_seed{seed}"
            result = read_json(run_dir / "result.json")
            if result["method"] != method or result["seed"] != seed or result["protocol_hash"] != protocol_hash:
                raise ValueError("Run identity or protocol mismatch")
            if result["augmentation_plan_sha256"] != plan:
                raise ValueError("Augmentation plan hash failed independent reconstruction")
            hashes.append(result["initial_state_sha256"])
            vp, vy, val = verify_predictions(run_dir / "val_predictions.npz", result["val"], expected["val"], W, polarity)
            p, y, test = verify_predictions(run_dir / "predictions.npz", result["test"], expected["test"], W, polarity)
            history = read_json(run_dir / "history.json")
            if [h["epoch"] for h in history] != list(range(1, max_epochs + 1)):
                raise ValueError("Incomplete or out-of-order history")
            best_index = int(np.argmax([h["val"]["ecc"] for h in history]))
            if result["best_epoch"] != best_index + 1 or not np.isclose(val["ecc"], history[best_index]["val"]["ecc"], atol=1e-6):
                raise ValueError("Checkpoint not selected by maximum validation ECC")
            if not np.isclose(result["best_validation_ecc"], val["ecc"], atol=1e-6):
                raise ValueError("Best validation score mismatch")
            runs[(method, seed)] = result
            predictions[(method, seed)] = p
            histories[(method, seed)] = history
        if len(set(hashes)) != 1 or any(not isinstance(x, str) or not re.fullmatch("[0-9a-f]{64}", x) for x in hashes):
            raise ValueError("Paired initial state hashes differ or are malformed")
    method_summary = {}
    for method in methods:
        method_summary[method] = {}
        for metric in METRICS:
            vals = np.array([runs[(method, seed)]["test"][metric] for seed in seeds])
            method_summary[method][metric] = {"mean": float(vals.mean()), "sample_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None}
    comparison_results = {}
    for method, baseline in comparisons:
        key = f"{method}_minus_{baseline}"
        comp = {"role": "primary" if baseline == "ce" else "secondary mechanistic control", "metrics": {}}
        for metric in METRICS:
            a = [runs[(baseline, seed)]["test"][metric] for seed in seeds]
            b = [runs[(method, seed)]["test"][metric] for seed in seeds]
            comp["metrics"][metric] = paired_seed_summary(a, b)
        differences = []
        for seed in seeds:
            a = predictions[(baseline, seed)].argmax(1)
            b = predictions[(method, seed)].argmax(1)
            differences.append(100 / W[y, b] - 100 / W[y, a])
        comp["ecc_crossed_bootstrap"] = crossed_seed_image_bootstrap(np.array(differences), y, repeats=repeats)
        comp["minimum_meaningful_ecc_gain"] = meaningful
        comparison_results[key] = comp
    result = {"dataset": directory.name, "seeds": seeds, "n_seed_pairs": len(seeds),
              "split_counts": {s: len(z) for s, z in expected.items()}, "classes": classes,
              "method_summary": method_summary, "comparisons": comparison_results,
              "protocol_hash": protocol_hash, "epochs": max_epochs}
    batch_size = protocol["args"].get("batch_size")
    result["planned_training_minibatches_per_run"] = int(max_epochs * np.ceil(len(expected["train"]) / batch_size)) if batch_size else None
    result["diagnostics"] = secondary_diagnostics(predictions, expected["test"], runs, seeds, classes, W, methods, comparisons)
    result["data_provenance"] = provenance
    verification = {"dataset": directory.name, "verified_runs": len(runs),
                    "verified_prediction_arrays": len(runs) * 2,
                    "metrics_recomputed": [*METRICS, "dirichlet", "error_energy"], "csv_labels_in_source_order": True,
                    "probabilities_finite_normalized": True,
                    "paired_initial_state_hashes_agree": True,
                    "initial_state_note": "Recorded hashes agree across methods within each seed; initial states are not independently regenerated by the report",
                    "augmentation_plan_hashes_independently_reconstructed": True,
                    "validation_checkpoint_selection_verified": True,
                    "protocol_and_executed_runner_hash_verified": True,
                    "executed_source_snapshot_hashes_verified": snapshot_hashes,
                    "snapshot_note": "Hashes verified against executed source snapshots, not current repository files",
                    "dataset_preparation_manifest_hash_verified": provenance["manifest_sha256"],
                    "parseval_brier_and_gsee_mode_identity_verified": True,
                    "training_scope_annotation": "Full-backbone fine-tuning: the executed runner optimizes model.parameters() with all backbone parameters trainable. " + ("The raw initialization phrase 'fixed backbone' refers to common starting weights, not a frozen feature extractor. " if "fixed backbone" in protocol.get("initialization", "") else "The common pretrained initialization does not freeze model parameters. ") + "Raw protocol/source are preserved unchanged.",
                    "planned_training_minibatches_per_run": result["planned_training_minibatches_per_run"],
                    "training_budget_note": "Equal method budgets within a dataset; the same epoch count across different dataset sizes implies different minibatch counts. Planned minibatches do not diagnose possible AMP skipped steps.",
                    "image_note": "CSV identities verified; report does not independently rehash every source image"}
    return result, verification, runs, histories


def provenance_markdown(datasets):
    lines = ["## Dataset provenance and data-quality findings", "",
             "Data quality was audited during preparation, before the FI model outcomes. These are fixed custom capped/split datasets with documented sources; neither a capped subset nor a nonstandard split is an official full-benchmark reproduction.", ""]
    for name, ds in datasets.items():
        manifest = ds["data_provenance"]["manifest_verbatim"]
        cleanup = manifest.get("ambiguity_cleanup", {})
        if cleanup:
            members = cleanup.get("raw_ambiguous_members_excluded", manifest.get("rejected_counts", {}).get("ambiguous_crosslabel_pixel_group"))
            lines += [f"**{name} placeholder correction before training.** The preparation audit found a decoded-pixel group with conflicting emotion labels: {members:,} raw members shared a placeholder image. Every member of the ambiguous group was excluded before split assignment, replacing the earlier first-label duplicate policy. The original resized-input diagnostic counted {manifest.get('identical_resized_input_diagnostic', 'unrecorded')} repeated rows; that count and the ambiguous-group membership overlap and must not be added together. The final rejection categories and retained counts are preserved verbatim in dataset_provenance.json. Original artifacts remain in `{cleanup.get('backup_directory', 'the preparation backup directory')}`. This correction used image identity and label consistency rather than model performance.", ""]
    lines += ["| Dataset | Raw archive rows / selected-shard availability | Usable before caps | Actual train | Actual validation | Actual test |", "|---|---:|---:|---:|---:|---:|"]
    for name, ds in datasets.items():
        summary = ds["data_provenance"]["summary"]
        split = summary["actual_split_counts"]
        raw = f"{summary['raw_rows']:,}" if summary["raw_rows"] is not None else "not recorded"
        if "selected mirror shards" in summary["raw_scope"]:
            raw += " (selected shards only)"
        usable = f"{summary['usable_before_caps']:,}" if summary["usable_before_caps"] is not None else "not recorded"
        lines.append(f"| {name} | {raw} | {usable} | {split['train']:,} | {split['val']:,} | {split['test']:,} |")
    imbalanced = [name for name, ds in datasets.items()
                  if any(len(set(counts)) > 1 for counts in ds["data_provenance"]["summary"]["actual_class_counts"].values())]
    balance_note = "Datasets with unequal class counts in at least one split: " + ", ".join(imbalanced) + "." if imbalanced else "The reported split counts happen to be balanced across classes."
    lines += ["", "Per-class caps are maxima, not a guarantee of perfect class balance. The verified actual CSV counts govern the experiment. " + balance_note + " The older EmoSet manifest records available shard rows and selected counts but does not record all usable pre-cap image counts, so that value is left unknown.", ""]
    for name, ds in datasets.items():
        provenance = ds["data_provenance"]; summary = provenance["summary"]
        lines += [f"### {name}: source and preprocessing", "",
                  f"- Mirror: `{summary['mirror']}`; pinned revision: `{summary['pinned_revision']}`.",
                  f"- Source: [{summary['source_url']}]({summary['source_url']})." if summary["source_url"] else "- Source URL not recorded.",
                  f"- Raw-count scope: {summary['raw_scope']}.",
                  f"- Prepared images: {summary['prepared_image_transform']} ({summary['preparation_description_source']}).",
                  f"- Classifier input pipeline: {summary['classifier_input_pipeline']}.",
                  f"- Split rule: {summary['split_rule']}.",
                  f"- Actual class counts in protocol class order {ds['classes']}: train {summary['actual_class_counts']['train']}; validation {summary['actual_class_counts']['val']}; test {summary['actual_class_counts']['test']}.",
                  f"- Exclusion limits: {summary['exclusion_limits']}.",
                  f"- Verified preparation manifest SHA256: `{provenance['manifest_sha256']}`."]
        if summary["archive_sha256"]:
            lines += [f"- Archive `{summary['archive']}` SHA256: `{summary['archive_sha256']}`."]
        for shard in summary["shards"]:
            lines += [f"- Source shard `{shard.get('path')}` SHA256: `{shard.get('sha256')}`."]
        lines += [""]
    lines += ["The report verifies the preparation manifest against `protocol.data_manifest_sha256` and preserves the complete original schema in [dataset_provenance.json](dataset_provenance.json). Archive/shard hashes are provenance recorded by preparation; this report does not re-download or independently rehash all source archives. Historical exclusion-source records can refer to preserved preparation snapshots rather than subsequently corrected CSV files. Exact-hash exclusion cannot rule out re-encoded, resized or near-duplicate overlap, and different emotion labels remain subjective.", ""]
    return lines


def prior_audit(repo):
    """Historical results are descriptive context; never pool their seeds."""
    audit = {}
    locations = {"full_emoset_8epoch_kaggle": repo / "data/kaggle_full_8epoch/emoset_graph_alpha_sweep_2gpu/runs",
                 "old_capped_emoset_finetune": repo / "outputs/graph_experiment/finetune",
                 "old_capped_emoset_probe": repo / "outputs/graph_experiment/probe"}
    for name, directory in locations.items():
        metric_file = "metrics.json" if "kaggle" in name else "result.json"
        graph = "graph_a0p3" if "kaggle" in name else "graph"
        pairs = []
        for seed in range(3):
            a = directory / f"ce_seed{seed}" / metric_file
            b = directory / f"{graph}_seed{seed}" / metric_file
            if a.exists() and b.exists():
                pairs.append((read_json(a), read_json(b)))
        if pairs:
            audit[name] = {"seed_pairs": list(range(len(pairs))), "graph_method": graph,
                           "selection_note": "Historical exploratory protocol; alpha0.3 selected on prior full-data validation; not fresh confirmatory seeds",
                           "metrics": {metric: paired_seed_summary([a["test"][metric] for a,b in pairs], [b["test"][metric] for a,b in pairs]) for metric in METRICS}}
    return audit


def conclusion(comp, meaningful):
    ecc = comp["metrics"]["ecc"]
    interval = ecc["t95"]
    delta = ecc["mean_difference"]
    text = f"Observed mean ECC difference is {delta:+.3f} points. "
    if interval is None:
        text += "Seed uncertainty could not be estimated by the Student interval. "
    elif interval[0] > 0:
        text += "The seed-level Student interval excludes zero in the positive direction. "
    elif interval[1] < 0:
        text += "The seed-level Student interval excludes zero in the negative direction. "
    else:
        text += "The seed-level Student interval includes zero, so these runs leave the benefit uncertain. "
    p = comp.get("ecc_holm_adjusted_p")
    if p is not None:
        text += f"Exact sign-flip Holm-adjusted p={p:.4g}; "
        text += "the familywise test rejects at 0.05. " if p < .05 else "the familywise test does not reject at 0.05. "
    text += f"The prespecified practical reference is +{meaningful:g} ECC point: "
    if interval is not None and interval[0] > meaningful:
        text += "the Student interval lies above that reference."
    else:
        text += "these results do not establish a gain exceeding that reference."
    return text


def make_figures(out, datasets, all_runs, all_histories):
    names = list(datasets)
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.5 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        ds, runs = datasets[name], all_runs[name]
        seeds = ds["seeds"]
        a = np.array([runs[("ce", s)]["test"]["ecc"] for s in seeds])
        b = np.array([runs[("graph", s)]["test"]["ecc"] for s in seeds])
        ax = axes[row, 0]
        ax.scatter(a, b)
        for x, y, seed in zip(a, b, seeds): ax.annotate(str(seed), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
        lo, hi = min(a.min(), b.min()) - .1, max(a.max(), b.max()) + .1
        ax.plot([lo, hi], [lo, hi], "--", color="gray")
        ax.set(xlabel="CE ECC (%)", ylabel="Graph ECC (%)", title=f"{name}: paired training seeds")
        ax = axes[row, 1]
        ax.scatter(b - a, np.arange(len(seeds)))
        ax.axvline(0, color="gray", linestyle="--")
        ax.axvline(.5, color="orange", linestyle=":", label="+0.5 reference")
        ax.set(yticks=np.arange(len(seeds)), yticklabels=seeds, xlabel="Graph − CE ECC points", ylabel="Training seed", title="Seed differences, fixed test split")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "paired_seeds.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(len(names), 1, figsize=(10, 3.8 * len(names)), squeeze=False)
    common_handles = {}
    for row, name in enumerate(names):
        ax = axes[row, 0]
        comps = datasets[name]["comparisons"]
        for i, (key, comp) in enumerate(comps.items()):
            t = comp["metrics"]["ecc"]
            mean, interval = t["mean_difference"], t["t95"]
            if interval is not None:
                handle = ax.errorbar(mean, i + .07, xerr=[[mean - interval[0]], [interval[1] - mean]], fmt="o", color="C0", capsize=4)
                common_handles.setdefault("Seed t95", handle)
            cross = comp["ecc_crossed_bootstrap"]["ci95"]
            handle = ax.errorbar(mean, i - .07, xerr=[[mean - cross[0]], [cross[1] - mean]], fmt="s", color="C1", capsize=4)
            common_handles.setdefault("Crossed seed/image 95%", handle)
        common_handles.setdefault("Zero difference", ax.axvline(0, linestyle="--", color="gray"))
        common_handles.setdefault("+0.5 practical reference", ax.axvline(.5, linestyle=":", color="orange"))
        ax.set(yticks=range(len(comps)), yticklabels=[k.replace("_", " ") for k in comps], xlabel="ECC difference (percentage points)", title=f"{name}: effect and uncertainty")
    fig.legend(list(common_handles.values()), list(common_handles), loc="upper center", bbox_to_anchor=(.5, .995), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - .8 / fig.get_size_inches()[1]))
    fig.savefig(out / "control_intervals.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 4.5), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0, j]
        for method in METHODS:
            curves = np.array([[h["val"]["ecc"] for h in all_histories[name][(method, seed)]] for seed in datasets[name]["seeds"]])
            x = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(0)
            line = ax.plot(x, mean, label=method)[0]
            sd = curves.std(0, ddof=1) if len(curves) > 1 else np.zeros(curves.shape[1])
            ax.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=.12)
        ax.set(xlabel="Epoch", ylabel="Validation ECC (%)", title=f"{name}: mean ± sample seed SD")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "validation_curves.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 4.7), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[0, j]
        diagnostic = datasets[name]["diagnostics"]
        upper_energy = 0.0
        for method in METHODS:
            spectrum = diagnostic["non_dc_eigenspace_energy"][method]
            x = [group["eigenvalue"] for group in spectrum]
            means = [group["mean"] for group in spectrum]
            sd = [group["sample_sd"] or 0 for group in spectrum]
            upper_energy = max(upper_energy, max(m + s for m, s in zip(means, sd)))
            ax.errorbar(x, means, yerr=sd, marker="o", markersize=4, capsize=2, label=method)
        ax.set(xlabel="Correct-graph Laplacian eigenvalue λ", ylabel="Mean residual energy per eigenspace",
               title=f"{name}: graph error energy, mean ± seed SD")
        ax.set_ylim(0, max(upper_energy, 1e-8) * 1.1)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "error_energy.png", dpi=160); plt.close(fig)


def write_diagnostic_csvs(out, datasets):
    files = {
        "per_class_seed_metrics.csv": ("per_class_seed_metrics", ["method", "seed", "class", "count", "acc", "ecc"]),
        "per_class_seed_differences.csv": ("per_class_seed_differences", ["comparison", "seed", "class", "count", "acc", "ecc"]),
    }
    for filename, (key, columns) in files.items():
        with (out / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["dataset", *columns])
            for name, ds in datasets.items():
                for row in ds["diagnostics"][key]: writer.writerow([name, *[row[c] for c in columns]])
    for filename, key, identifier in (("per_class_summary.csv", "per_class_summary", "method"),
                                       ("per_class_difference_summary.csv", "per_class_difference_summary", "comparison")):
        with (out / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(["dataset", identifier, "class", "count", "acc_mean", "acc_seed_sd", "ecc_mean", "ecc_seed_sd"])
            for name, ds in datasets.items():
                for row in ds["diagnostics"][key]:
                    writer.writerow([name, row[identifier], row["class"], row["count"], row["acc"]["mean"], row["acc"]["sample_sd"], row["ecc"]["mean"], row["ecc"]["sample_sd"]])
    with (out / "error_energy.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["dataset", "method", "mode_indices", "multiplicity", "eigenvalue", "energy_mean", "energy_seed_sd"])
        for name, ds in datasets.items():
            for method, spectrum in ds["diagnostics"]["non_dc_eigenspace_energy"].items():
                for group in spectrum:
                    writer.writerow([name, method, ";".join(map(str, group["indices"])), group["multiplicity"], group["eigenvalue"], group["mean"], group["sample_sd"]])
    with (out / "probability_diagnostics.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["dataset", "method", "metric", "mean", "sample_seed_sd"])
        for name, ds in datasets.items():
            diagnostic = ds["diagnostics"]
            for method in diagnostic["probability_roughness"]:
                metrics = dict(diagnostic["probability_roughness"][method])
                metrics["balanced_macro_ecc"] = diagnostic["balanced_macro_ecc"][method]
                for key, stats in metrics.items(): writer.writerow([name, method, key, stats["mean"], stats["sample_sd"]])


def report(args):
    out, repo = args.out.resolve(), args.repo.resolve()
    directories = [out] if (out / "protocol.json").exists() else sorted(p for p in out.iterdir() if p.is_dir() and (p / "protocol.json").exists())
    if not directories:
        raise ValueError("No completed dataset study directories found")
    datasets, verification, all_runs, all_histories = {}, {}, {}, {}
    for directory in directories:
        ds, ver, runs, histories = analyze_dataset(directory, repo, args.bootstrap, args.meaningful_ecc)
        datasets[directory.name], verification[directory.name] = ds, ver
        all_runs[directory.name], all_histories[directory.name] = runs, histories
    family = [comp for ds in datasets.values() for comp in ds["comparisons"].values()]
    pvalues = [comp["metrics"]["ecc"]["exact_signflip"]["p_two_sided"] for comp in family]
    for comp, adjusted in zip(family, holm(pvalues)):
        comp["ecc_holm_adjusted_p"] = adjusted
        comp["conclusion"] = conclusion(comp, args.meaningful_ecc)
    prior = prior_audit(repo)
    summary = {"datasets": datasets, "prior_studies_not_pooled": prior,
               "primary_endpoint": "ECC graph minus CE separately within each dataset",
               "multiplicity": f"Holm across all {len(family)} exact paired-seed ECC comparisons (three per dataset); other metrics descriptive",
               "meaningful_ecc_gain_reference": args.meaningful_ecc,
               "scope": "Repeatability across fresh initialization/augmentation seeds on fixed splits; not independent training datasets or human validation"}
    write_json(out / "summary.json", summary); write_json(out / "verification.json", verification)
    write_json(out / "prior_seed_audit.json", prior)
    write_json(out / "dataset_provenance.json", {name: ds["data_provenance"] for name, ds in datasets.items()})
    with (out / "summary.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["dataset", "method", "metric", "mean", "sample_seed_sd", "n_seeds"])
        for name, ds in datasets.items():
            for method, metrics in ds["method_summary"].items():
                for key, value in metrics.items(): writer.writerow([name, method, key, value["mean"], value["sample_sd"], ds["n_seed_pairs"]])
    with (out / "seed_results.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["dataset", "method", "seed", "best_epoch", *METRICS])
        for name, runs in all_runs.items():
            for (method, seed), run in runs.items(): writer.writerow([name, method, seed, run["best_epoch"], *[run["test"][k] for k in METRICS]])
    write_diagnostic_csvs(out, datasets)
    make_figures(out, datasets, all_runs, all_histories)
    lines = ["# Fresh paired-seed evidence for the emotion-label graph loss", "",
             "This study asks whether the observed ECC benefit repeats across training seeds and whether emotional geometry contributes beyond a matched generic quadratic. Each dataset is analyzed separately. No pooled dataset confidence interval is reported.", "",
             "## Locked comparison and signal-processing meaning", "",
             "CE, graph, isotropic and wrong-label-permuted graph use identical pretrained initialization and explicit crop/flip/order plans within each seed. These are full-backbone fine-tuning runs: the optimizer receives all model parameters. The raw protocol's phrase 'fixed backbone' describes the common starting ImageNet weights, not frozen backbone parameters; raw executed protocols and source snapshots are preserved unchanged. The coefficient stays alpha=0.3, gamma=1 and beta=0.01; no test-driven alpha choice is made. Epochs are selected only by validation ECC. Every method has the same training budget within its dataset.", "",
             "The signal is the eight emotion probabilities on a weighted label graph. The graph residual quadratic uses Q=L+ (the Laplacian pseudoinverse), emphasizing low graph-frequency residuals; the separate p^T L p term penalizes roughness. These graph frequencies are distinct from spatial image FFT frequencies. The isotropic control independently matches tr(Q) and tr(L), removing graph direction while matching average quadratic scales. The wrong-label control preserves the entire spectrum under a vertex permutation but breaks alignment with emotion labels. Its preassigned permutation varies by seed, so that comparison includes random-geometry variation as well as training variation.", ""]
    lines += provenance_markdown(datasets)
    lines += ["## New experiments", ""]
    for name, ds in datasets.items():
        lines += [f"### {name}", "", f"{ds['n_seed_pairs']} fresh seed pairs {ds['seeds']}; {ds['epochs']} epochs per run. Train/validation/test counts: {ds['split_counts']}. Four methods, {4 * ds['n_seed_pairs']} completed runs. Planned training minibatches per run: {ds['planned_training_minibatches_per_run']}.", "",
                  "| Method | ACC | ECC | ACC2 | EMC | Brier | GSEE |", "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            vals = ds["method_summary"][method]
            lines.append("| " + method + " | " + " | ".join(f"{vals[m]['mean']:.4f} ± {vals[m]['sample_sd']:.4f}" if vals[m]['sample_sd'] is not None else f"{vals[m]['mean']:.4f}" for m in METRICS) + " |")
        lines += ["", "Values are mean ± sample training-seed SD; ACC/ECC/ACC2/EMC are percentages, Brier/GSEE are raw energy errors. Lower Brier/GSEE is better.", ""]
        for key, comp in ds["comparisons"].items():
            s = comp["metrics"]["ecc"]; cross = comp["ecc_crossed_bootstrap"]
            interval = s["t95"]
            ci = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval is not None else "unavailable"
            lines += [f"**{key.replace('_', ' ')} ({comp['role']}).** {comp['conclusion']}", "",
                      f"Seed t95 {ci}; crossed seed/image percentile interval [{cross['ci95'][0]:+.3f}, {cross['ci95'][1]:+.3f}]. Positive/negative/zero pairs: {s['positive_pairs']}/{s['negative_pairs']}/{s['zero_pairs']}. Leave-one-seed-out mean range: {s['leave_one_out_mean_range']}.", ""]
    lines += ["![Paired seed outcomes](paired_seeds.png)", "", "![Control effect intervals](control_intervals.png)", "", "![Validation curves](validation_curves.png)", "",
              "## Secondary class and mode diagnostics", "",
              "Every model, including the wrong-label graph training control, is analyzed using the same correct emotion graph. Per-class accuracy and ECC are computed from all test images of each class, then summarized across paired training seeds. Class counts and seed SD accompany all means; these are descriptive analyses, without new class-wise significance tests or selecting favorable classes.", "",
              "The balanced macro-ECC diagnostic averages the eight class-wise ECC scores equally. Primary ECC weights each test image equally. They coincide on a balanced test split and can differ on an imbalanced one. Macro-ECC is not substituted for the prespecified primary endpoint.", ""]
    for name, ds in datasets.items():
        diagnostic = ds["diagnostics"]
        lines += [f"### {name}: probability diagnostics", "",
                  "| Method | Balanced macro-ECC | Probability roughness p^T L p | Brier | GSEE |", "|---|---:|---:|---:|---:|"]
        for method in METHODS:
            macro = diagnostic["balanced_macro_ecc"][method]
            rough = diagnostic["probability_roughness"][method]
            stats = [macro, rough["dirichlet"], rough["brier"], rough["gsee"]]
            lines.append("| " + method + " | " + " | ".join(f"{z['mean']:.4f} ± {z['sample_sd']:.4f}" if z['sample_sd'] is not None else f"{z['mean']:.4f}" for z in stats) + " |")
        lines += [""]
    lines += ["![Correct-graph error energy](error_energy.png)", "",
              "Modal residual energy is computed from the correct-graph GFT of p-y. Non-DC mode/eigenspace means and seed SD are shown. Repeated eigenvalues, if present, are grouped and their energies summed to make the diagnostic invariant to an arbitrary choice of eigenvectors within the repeated eigenspace. The present eight-class graph has distinct non-DC eigenvalues. The total modal energy equals Brier error by Parseval; inverse-eigenvalue weighting gives GSEE. These algebraic identities are also checked.", "",
              "GSEE and mode energy are directly related to the trained quadratic objective. Lower values or a visually favorable spectrum are not independent evidence of better recognition. Predicted-probability roughness is likewise a diagnostic: uniform predictions minimize it while discarding class information. Accuracy/ECC outcomes remain necessary.", "",
              "Full secondary tables: [per-class seed metrics](per_class_seed_metrics.csv), [per-class means and seed SD](per_class_summary.csv), [paired per-class seed differences](per_class_seed_differences.csv), [per-class difference summaries](per_class_difference_summary.csv), [modal energy](error_energy.csv), [probability diagnostics](probability_diagnostics.csv).", "",
              "## Historical results, kept separate", ""]
    for name, result in prior.items():
        s = result["metrics"]["ecc"]
        lines += [f"- {name}: {s['n_seed_pairs']} paired seeds, graph-minus-CE ECC {s['mean_difference']:+.4f}, differences {s['differences']}, seed t95 {s['t95']}, exact sign-flip p={s['exact_signflip']['p_two_sided']:.4g}."]
    lines += ["", "The original full EmoSet alpha0.3 gain was +0.268 ECC point from only three seeds. A positive image-bootstrap interval conditional on those checkpoints did not establish repeatability over future training seeds. Historical capped-subset, frozen-feature and full-data runs have different protocols and are never combined into one effect estimate. Historical metrics here are read from prior artifacts rather than independently rerun training.", "",
              "## Inference and limits", "",
              summary["multiplicity"] + ". Student t95 assumes independent seed pairs with approximately normal differences. Exact sign-flip p-values assume within-pair treatment-label exchangeability/sign symmetry under the null. The crossed bootstrap independently resamples seed pairs and images within each class, sharing each image draw across all seeds and methods. It is an exploratory percentile interval and cannot manufacture new training seeds.", "",
              f"The +{args.meaningful_ecc:g}-point reference is a prespecified practical target, not a claim of achieved statistical power. Failure to reject is not evidence of equivalence; no TOST equivalence test was prespecified. No claim that the graph is wrong follows from an uncertain difference. Multiple secondary metrics are descriptive.", "",
              "The same epoch count across datasets does not give the same optimization budget: at batch64, eight epochs give 600 planned training minibatches on the 4,800-image EmoSet subset and 64 on 468-image ArtPhoto. These are planned minibatch counts; AMP skipped steps are not separately diagnosed. Effect comparisons are within each dataset, and outcomes across datasets must not be interpreted as performance under equal numbers of optimization steps.", "",
              "The data split is fixed and previously inspected benchmark images remain reused test data; fresh training seeds do not make a fresh benchmark. Dataset-specific findings do not establish population-wide improvement. A resplit or capped subset of EmoSet is not an independent dataset. GSEE is directly related to the optimized loss, so lower GSEE is not independent validation. The graph itself comes from the scoring wheel, not a learned discovery about human brains.", "",
              "## Verification", "",
              "All validation/test probability arrays were checked for normalization, finite values, shape and exact CSV label order. ACC, ECC, ACC2, EMC, Brier, GSEE, probability roughness and every mode's residual energy were recomputed from saved probabilities. Dataset preparation manifest hashes, CSV hashes, protocol hash and executed runner/graph/metrics/emotion source snapshot hashes were verified; these use the archived source snapshots rather than current repository files. Complete histories and earliest maximum-validation-ECC checkpoint selection were checked. Recorded initialization hashes agree within paired methods; augmentation plan hashes were independently reconstructed. Full details are in verification.json. Source image byte hashes and model training itself were not independently rerun by this report.", ""]
    markdown_text = "\n".join(lines)
    (out / "REPORT.md").write_text(markdown_text, encoding="utf8")
    try:
        import markdown
        body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + html.escape(markdown_text) + "</pre>"
    for filename in ("paired_seeds.png", "control_intervals.png", "validation_curves.png", "error_energy.png"):
        uri = "data:image/png;base64," + base64.b64encode((out / filename).read_bytes()).decode()
        if "<pre>" in body:
            body += f'<img alt="{filename}" src="{uri}">'
        else:
            body = body.replace(f'src="{filename}"', f'src="{uri}"')
    document = '<!doctype html><html><head><meta charset="utf-8"><title>Paired training-seed evidence</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 25px;color:#18202a}img{max-width:100%}table{border-collapse:collapse;font-size:14px}th,td{padding:7px;border:1px solid #ddd}th{background:#edf3fa}pre{white-space:pre-wrap}</style></head><body>' + body + '</body></html>'
    (out / "REPORT.html").write_text(document, encoding="utf8")
    print(json.dumps({"datasets": list(datasets), "verified_runs": sum(v["verified_runs"] for v in verification.values()), "report": str(out / "REPORT.html")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("outputs/seed_evidence"))
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--bootstrap", type=int, default=1500)
    p.add_argument("--meaningful-ecc", type=float, default=.5)
    args = p.parse_args()
    if args.bootstrap < 100 or not np.isfinite(args.meaningful_ecc) or args.meaningful_ecc <= 0:
        p.error("Require bootstrap>=100 and finite positive practical reference")
    report(args)


if __name__ == "__main__":
    main()


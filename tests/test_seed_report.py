"""Small deterministic saved-artifact fixture; no training/GPU needed."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

try:
    import ecc.seed_statistics
except ImportError:
    import seed_statistics
    sys.modules["ecc.seed_statistics"] = seed_statistics

REPORT_PATH = Path(__file__).with_name("report_seed_evidence.py")
if not REPORT_PATH.exists():
    REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts/report_seed_evidence.py"
spec = importlib.util.spec_from_file_location("seed_report", REPORT_PATH)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def make_fixture(tmp_path):
    data, out = tmp_path / "data", tmp_path / "results"
    data.mkdir(); out.mkdir()
    classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    y = np.arange(16) % 8
    p = np.full((len(y), 8), .1 / 7)
    p[np.arange(len(y)), y] = .9
    W = reporter.build_cost_matrix(classes)
    pol = reporter.polarity_vector(classes)
    metrics = reporter.evaluate_predictions(y, p.argmax(1), W, pol)
    metrics.update(reporter.graph_error_metrics(p, y, W))
    metadata = {}
    for split in ("train", "val", "test"):
        manifest = data / f"{split}.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label"])
            writer.writerows((f"{i}.jpg", classes[c]) for i, c in enumerate(y))
        metadata[split] = {"n": len(y), "csv_sha256": reporter.sha256(manifest.read_bytes()), "class_counts": [2] * 8}
    source = b"# fixture only; no model training\n"
    (out / "runner_source.py").write_bytes(source)
    sources = {"runner": reporter.sha256(source)}
    for name in ("graph", "metrics", "emotion"):
        contents = f"# {name}: executed-source fixture, deliberately different from current repository\n".encode()
        (out / f"{name}_source.py").write_bytes(contents)
        sources[f"ecc.{name}"] = reporter.sha256(contents)
    manifest = {"dataset": "fixture", "raw_counts": {c: 2 for c in classes}, "retained_counts": {c: 2 for c in classes},
                "mirror": "example/fixture", "revision": "fixed-fixture-revision", "source_url": "https://example.test/fixture",
                "image_transform": "fixture only", "split_rule": "fixture only", "exclusion_scope": "fixture only"}
    reporter.write_json(data / "manifest.json", manifest)
    protocol = {"args": {"data_root": str(data), "epochs": 2, "batch_size": 8, "seeds": [3, 4, 5], "methods": list(reporter.METHODS)},
                "classes": classes, "datasets": metadata, "sources": sources,
                "initialization": "ImageNet1K ResNet18 fixed backbone plus seeded linear head",
                "data_manifest_sha256": reporter.sha256((data / "manifest.json").read_bytes())}
    protocol["protocol_hash"] = reporter.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode())
    reporter.write_json(out / "protocol.json", protocol)
    for seed in (3, 4, 5):
        for method in reporter.METHODS:
            directory = out / f"{method}_seed{seed}"
            directory.mkdir()
            result = {"method": method, "seed": seed, "protocol_hash": protocol["protocol_hash"],
                      "augmentation_plan_sha256": reporter.plan_digest(len(y), 2, seed),
                      "initial_state_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
                      "val": metrics, "test": metrics, "best_epoch": 1, "best_validation_ecc": metrics["ecc"]}
            reporter.write_json(directory / "result.json", result)
            reporter.write_json(directory / "history.json", [{"epoch": i, "train_loss": 1., "val": metrics} for i in (1, 2)])
            for filename in ("predictions.npz", "val_predictions.npz"):
                np.savez_compressed(directory / filename, probabilities=p, labels=y)
    return out, tmp_path


def test_holm_known_order_and_bounds():
    assert reporter.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert reporter.holm([1, 0]) == [1, 0]
    with pytest.raises(ValueError):
        reporter.holm([np.nan])


def test_complete_artifact_validation_and_tamper_detection(tmp_path):
    out, repo = make_fixture(tmp_path)
    summary, verification, _, _ = reporter.analyze_dataset(out, repo, 100, .5)
    assert verification["verified_runs"] == 12
    assert verification["planned_training_minibatches_per_run"] == 4
    assert "not a frozen" in verification["training_scope_annotation"]
    assert summary["comparisons"]["graph_minus_ce"]["metrics"]["ecc"]["mean_difference"] == 0
    assert summary["comparisons"]["graph_minus_ce"]["ecc_crossed_bootstrap"]["ci95"] == [0, 0]
    wrong = reporter.read_json(out / "graph_seed3/result.json")
    wrong["initial_state_sha256"] = "0" * 64
    reporter.write_json(out / "graph_seed3/result.json", wrong)
    with pytest.raises(ValueError, match="initial state"):
        reporter.analyze_dataset(out, repo, 100, .5)


def test_probability_metric_tampering_rejected(tmp_path):
    out, repo = make_fixture(tmp_path)
    wrong = reporter.read_json(out / "graph_seed3/result.json")
    wrong["test"]["ecc"] -= .01
    reporter.write_json(out / "graph_seed3/result.json", wrong)
    with pytest.raises(ValueError, match="Metric mismatch ecc"):
        reporter.analyze_dataset(out, repo, 100, .5)


def test_executed_source_snapshot_tampering_rejected(tmp_path):
    out, repo = make_fixture(tmp_path)
    (out / "graph_source.py").write_text("# altered archived graph source\n")
    with pytest.raises(ValueError, match="source snapshot mismatch"):
        reporter.analyze_dataset(out, repo, 100, .5)


def test_preparation_manifest_tampering_rejected(tmp_path):
    out, repo = make_fixture(tmp_path)
    (repo / "data/manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        reporter.analyze_dataset(out, repo, 100, .5)


def test_older_manifest_schema_is_preserved_without_inventing_usable_counts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    manifest = {s: {"available_rows": n, "selected": n // 2, "per_class": {"awe": n // 2}, "excluded": {}}
                for s, n in zip(("train", "val", "test"), (100, 20, 40))}
    manifest["source"] = {"repository": "udeman/EmoSet-118K", "revision": "pinned", "files": [{"path": "data/train.parquet", "sha256": "0" * 64}]}
    reporter.write_json(data / "manifest.json", manifest)
    protocol = {"data_manifest_sha256": reporter.sha256((data / "manifest.json").read_bytes()),
                "datasets": {s: {"n": n, "class_counts": [n]} for s, n in (("train", 50), ("val", 10), ("test", 20))}}
    result = reporter.dataset_provenance(data, protocol)
    assert result["manifest_verbatim"] == manifest
    assert result["summary"]["raw_rows"] == 160
    assert result["summary"]["usable_before_caps"] is None
    assert result["summary"]["source_url"].endswith("/tree/pinned")
    assert result["summary"]["shards"] == manifest["source"]["files"]


def test_saved_mode_energy_tampering_rejected(tmp_path):
    out, repo = make_fixture(tmp_path)
    wrong = reporter.read_json(out / "graph_seed3/result.json")
    wrong["test"]["error_energy"][1] += .01
    reporter.write_json(out / "graph_seed3/result.json", wrong)
    with pytest.raises(ValueError, match="error_energy"):
        reporter.analyze_dataset(out, repo, 100, .5)


def test_per_class_macro_ecc_is_distinct_on_unbalanced_split():
    classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    W, pol = reporter.build_cost_matrix(classes), reporter.polarity_vector(classes)
    y = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    p = np.eye(8)[np.zeros(len(y), dtype=int)]
    metrics = reporter.evaluate_predictions(y, p.argmax(1), W, pol)
    metrics.update(reporter.graph_error_metrics(p, y, W))
    seeds = [3, 4]
    runs = {(m, s): {"test": metrics} for m in reporter.METHODS for s in seeds}
    predictions = {(m, s): p for m in reporter.METHODS for s in seeds}
    result = reporter.secondary_diagnostics(predictions, y, runs, seeds, classes, W)
    expected_macro = float((100 / W[np.arange(8), 0]).mean())
    assert result["balanced_macro_ecc"]["ce"]["mean"] == pytest.approx(expected_macro)
    assert not np.isclose(expected_macro, metrics["ecc"])
    assert len(result["per_class_seed_metrics"]) == 4 * 2 * 8
    assert all(row["ecc"] == 0 and row["acc"] == 0 for row in result["per_class_seed_differences"])
    class_zero = next(row for row in result["per_class_summary"] if row["method"] == "ce" and row["class"] == classes[0])
    assert class_zero["count"] == 3
    assert class_zero["acc"]["mean"] == 100
    assert class_zero["acc"]["sample_sd"] == 0
    assert len(result["non_dc_eigenspace_energy"]["ce"]) == 7


def test_repeated_eigenvalues_are_grouped():
    assert reporter.eigenspace_groups([0, 1, 1 + 1e-12, 2]) == [[1, 2], [3]]


def test_conclusion_does_not_confuse_nonrejection_and_equivalence():
    stats = reporter.paired_seed_summary([10, 20, 30], [9, 22, 30])
    comp = {"metrics": {"ecc": stats}, "ecc_holm_adjusted_p": 1.0}
    text = reporter.conclusion(comp, .5)
    assert "includes zero" in text
    assert "does not reject" in text
    assert "do not establish" in text


def test_standalone_html_and_csv_generation(tmp_path):
    from argparse import Namespace
    out, repo = make_fixture(tmp_path)
    reporter.report(Namespace(out=out, repo=repo, bootstrap=100, meaningful_ecc=.5))
    document = (out / "REPORT.html").read_text(encoding="utf8")
    assert "data:image/png;base64," in document
    assert (out / "summary.csv").exists()
    assert (out / "seed_results.csv").exists()
    assert (out / "error_energy.png").exists()
    assert (out / "per_class_difference_summary.csv").exists()
    assert (out / "probability_diagnostics.csv").exists()
    assert reporter.read_json(out / "dataset_provenance.json")[out.name]["manifest_hash_verified"]
    assert "macro-ECC" in document
    assert reporter.read_json(out / "verification.json")[out.name]["verified_runs"] == 12

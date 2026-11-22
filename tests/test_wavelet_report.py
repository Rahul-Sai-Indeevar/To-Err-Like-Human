"""Saved-array fixture tests for the optional wavelet report; no training/GPU."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPORT = Path(__file__).resolve().parents[1] / "scripts/report_wavelet_study.py"
spec = importlib.util.spec_from_file_location("wavelet_report", REPORT)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def hashed(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "data/fixture"
    out = repo / "outputs/wavelet_study"
    dataset = out / "fixture"
    data.mkdir(parents=True)
    dataset.mkdir(parents=True)
    classes = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
    (data / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf8")
    reporter.write_json(data / "manifest.json", {"source": {"repository": "synthetic fixture only"}})
    source = repo / "scripts/fixture_runner.py"
    source.parent.mkdir()
    source.write_text("# fixture, no training\n", encoding="utf8")
    archive = dataset / "sources/scripts/fixture_runner.py"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(source.read_bytes())
    plan = repo / "docs/wavelet_study/STUDY_PLAN.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Fixture study plan\n", encoding="utf8")
    labels = np.arange(16) % 8
    metadata, cache_hashes = {}, {}
    for split in ("train", "val", "test"):
        path = data / f"{split}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "label"])
            writer.writerows((f"{i}.jpg", classes[y]) for i, y in enumerate(labels))
        key = hashlib.sha256(path.read_bytes() + b"PIL_EXIF_RGB_torchvisionResize192_antialias_v1").hexdigest()
        cache = data / f"{split}_seed_evidence_{key[:16]}.npy"
        np.save(cache, np.zeros((len(labels), 3, 192, 192), dtype=np.uint8))
        metadata[split] = {"n": len(labels), "csv_sha256": reporter.file_digest(path), "cache_key": key, "class_counts": [2] * 8}
        cache_hashes[split] = reporter.file_digest(cache)
    protocol = {"args": {"data_root": "data/fixture", "epochs": 2, "seeds": [15, 16, 17],
                          "methods": list(reporter.METHODS), "levels": 2, "batch_size": 8},
                "classes": classes, "datasets": metadata,
                "sources": {"scripts/fixture_runner.py": reporter.file_digest(source)},
                "cache_sha256": cache_hashes,
                "study_plan_path": "docs/wavelet_study/STUDY_PLAN.md", "study_plan_sha256": reporter.file_digest(plan),
                "data_manifest_sha256": reporter.file_digest(data / "manifest.json")}
    protocol["protocol_hash"] = hashlib.sha256(reporter.canonical(protocol)).hexdigest()
    reporter.write_json(dataset / "protocol.json", protocol)
    W, polarity = reporter.build_cost_matrix(classes), reporter.polarity_vector(classes)
    for seed in protocol["args"]["seeds"]:
        for method in reporter.METHODS:
            directory = dataset / f"{method}_seed{seed}"
            directory.mkdir()
            probabilities = np.full((len(labels), 8), .1 / 7)
            probabilities[np.arange(len(labels)), labels] = .9
            metrics = reporter._helper.evaluate_predictions(labels, probabilities.argmax(1), W, polarity)
            metrics.update(reporter._helper.graph_error_metrics(probabilities, labels, W))
            group = "single" if method in ("rgb", "low", "detail") else "dual"
            (directory / "best.pt").write_bytes(f"checkpoint fixture {method}/{seed}".encode())
            result = {"method": method, "seed": seed, "status": "evaluated", "protocol_hash": protocol["protocol_hash"],
                      "best_epoch": 1, "best_validation_ecc": metrics["ecc"], "seconds": 2.,
                      "initial_backbone_sha256": hashed(seed), "initial_model_sha256": hashed((seed, group)),
                      "initial_classifier_sha256": hashed((seed, group, "classifier")),
                      "augmentation_plan_sha256": reporter.plan_digest(len(labels), 2, seed),
                      "parameter_count": 10 if group == "single" else 20,
                      "optimizer_steps": 4, "amp_skipped_steps": 0,
                      "checkpoint_sha256": reporter.file_digest(directory / "best.pt"), "val": metrics, "test": metrics}
            reporter.write_json(directory / "result.json", result)
            reporter.write_json(directory / "history.json", [{"epoch": epoch, "train_loss": 1., "val": metrics, "seconds": 1.,
                                                                 "optimizer_steps": 2, "amp_skipped_steps": 0} for epoch in (1, 2)])
            for filename in ("predictions.npz", "val_predictions.npz"):
                np.savez_compressed(directory / filename, probabilities=probabilities, labels=labels)
    return repo, out, dataset


def test_full_verification_and_capacity_contrasts(tmp_path):
    repo, _, data = fixture(tmp_path)
    summary, audit, rows, classes = reporter.analyze_dataset(data, repo, bootstrap=100)
    assert audit["verified_runs"] == 18
    assert audit["prediction_arrays_verified"] == 36
    assert audit["checkpoints_hashed"] == 18
    assert audit["optimizer_steps"] == 72
    assert audit["amp_skipped_steps"] == 0
    assert len(rows) == 18 and len(classes) == 18 * 8
    assert summary["comparisons"]["wavelet_fusion_minus_rgb_dual"]["family"] == "primary"
    assert len(summary["comparisons"]) == 5
    assert summary["comparisons"]["wavelet_fusion_minus_rgb_dual"]["ecc_crossed_bootstrap"]["ci95"] == [0., 0.]


@pytest.mark.parametrize(("field", "value", "message"), [
    ("initial_backbone_sha256", "bad", "Paired backbone"),
    ("initial_model_sha256", "bad", "Parameter-matched"),
    ("initial_classifier_sha256", "bad", "Parameter-matched"),
    ("augmentation_plan_sha256", "bad", "Augmentation plan"),
    ("parameter_count", 21, "Parameter-matched"),
    ("status", "trained", "Unfinished"),
    ("best_epoch", 2, "checkpoint selection"),
    ("optimizer_steps", 5, "Optimizer result/history"),
])
def test_result_tampering_is_rejected(tmp_path, field, value, message):
    repo, _, data = fixture(tmp_path)
    path = data / "wavelet_fusion_seed15/result.json"
    result = reporter.read_json(path)
    result[field] = value
    reporter.write_json(path, result)
    with pytest.raises(ValueError, match=message):
        reporter.analyze_dataset(data, repo, bootstrap=100)


def test_probability_metrics_and_label_tampering_rejected(tmp_path):
    repo, _, data = fixture(tmp_path)
    path = data / "wavelet_fusion_seed15/result.json"
    result = reporter.read_json(path)
    result["test"]["ecc"] -= .1
    reporter.write_json(path, result)
    with pytest.raises(ValueError, match="Metric mismatch ecc"):
        reporter.analyze_dataset(data, repo, bootstrap=100)


@pytest.mark.parametrize(("relative", "message"), [
    ("outputs/wavelet_study/fixture/sources/scripts/fixture_runner.py", "source snapshot"),
    ("scripts/fixture_runner.py", "Current source"),
    ("docs/wavelet_study/STUDY_PLAN.md", "Study plan"),
    ("data/fixture/manifest.json", "preparation manifest"),
])
def test_input_and_executed_source_fingerprints_reject_tampering(tmp_path, relative, message):
    repo, _, data = fixture(tmp_path)
    (repo / relative).write_text("altered", encoding="utf8")
    with pytest.raises(ValueError, match=message):
        reporter.analyze_dataset(data, repo, bootstrap=100)


def test_input_cache_change_rejected(tmp_path):
    repo, _, data = fixture(tmp_path)
    cache = next((repo / "data/fixture").glob("train_seed*.npy"))
    array = np.load(cache)
    array[0, 0, 0, 0] = 1
    np.save(cache, array)
    with pytest.raises(ValueError, match="input cache hash"):
        reporter.analyze_dataset(data, repo, bootstrap=100)


def test_protocol_change_rejected(tmp_path):
    repo, _, data = fixture(tmp_path)
    path = data / "protocol.json"
    protocol = reporter.read_json(path)
    protocol["args"]["levels"] = 3
    reporter.write_json(path, protocol)
    with pytest.raises(ValueError, match="Protocol hash"):
        reporter.analyze_dataset(data, repo, bootstrap=100)


def test_holm_families_are_separate_and_do_not_pool_seeds():
    def comparison(p, family):
        return {"family": family, "metrics": {"ecc": {"exact_signflip": {"p_two_sided": p}}}}
    datasets = {"a": {"comparisons": {"primary": comparison(.01, "primary"), "secondary": comparison(.02, "secondary")}},
                "b": {"comparisons": {"primary": comparison(.04, "primary"), "secondary": comparison(.03, "secondary")}}}
    families = reporter.adjust_families(datasets)
    assert [r["adjusted_p"] for r in families["primary"]] == pytest.approx([.02, .04])
    assert [r["adjusted_p"] for r in families["secondary"]] == pytest.approx([.04, .04])


def test_report_standalone_outputs_and_honest_nonrejection(tmp_path):
    repo, out, _ = fixture(tmp_path)
    summary = reporter.build_report(out, bootstrap=100, repo_root=repo)
    assert summary["total_models"] == 18
    assert len(summary["multiplicity_families"]["primary"]) == 1
    assert len(summary["multiplicity_families"]["secondary"]) == 4
    assert (out / "seed_results.csv").exists()
    assert (out / "per_class_seed_metrics.csv").exists()
    assert (out / "summary.csv").exists()
    assert "data:image/png;base64," in (out / "REPORT.html").read_text(encoding="utf8")
    text = (out / "REPORT.md").read_text(encoding="utf8")
    assert "does not establish equality" in text
    assert "replay model inference" in text
    assert "physical signal-to-noise" in text

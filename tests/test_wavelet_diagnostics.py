"""Independent saved-artifact diagnostics fixtures; no models or GPU execution."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wavelet_diagnostics", ROOT / "scripts/report_wavelet_diagnostics.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)
fixture_spec = importlib.util.spec_from_file_location("wavelet_base_fixture", Path(__file__).with_name("test_wavelet_report.py"))
base_fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(base_fixture)


def _class_summaries(rows, classes):
    keys = [key for key in rows[0] if key.endswith("energy_fraction")]
    return [{"class": name, "n": sum(row["class"] == name for row in rows),
             **{key: float(np.mean([row[key] for row in rows if row["class"] == name])) for key in keys}}
            for name in classes]


def fixture(tmp_path):
    repo, out, dataset = base_fixture.fixture(tmp_path)
    protocol = reporter.read_json(dataset / "protocol.json")
    methods, seeds, classes = protocol["args"]["methods"], protocol["args"]["seeds"], protocol["classes"]
    labels = np.arange(16) % 8
    image_energy = []
    for index, label in enumerate(labels):
        image_energy.append({"train_row": index, "class": classes[label],
                             "normalized_low_energy_fraction": .8,
                             "spatially_centered_low_energy_fraction": .5,
                             **{f"{band}_energy_fraction": .8 if band == "LL2" else .2 / 6 for band in reporter.BANDS},
                             **{f"centered_{band}_energy_fraction": .5 if band == "LL2" else .5 / 6 for band in reporter.BANDS}})
    transform = {"status": "passed", "input_scope": "synthetic saved-artifact fixture, not a real input audit",
                 "sample_count": len(labels), "selected_train_rows": list(range(len(labels))),
                 "maxima": {key: 1e-15 for key in reporter.IDENTITIES},
                 "shift1_relative_commutator_mean": .07,
                 "noise": {"distribution": "synthetic fixture", "expected_low_fraction": 1 / 16,
                           "expected_detail_fraction": 15 / 16, "measured_low_fraction": .0624, "measured_detail_fraction": .9376},
                 "image_energy": image_energy, "class_summary": _class_summaries(image_energy, classes)}
    reporter.write_json(dataset / "transform_diagnostics.json", transform)
    replay = {"status": "passed", "scope": "validation saved-artifact fixture only",
              "arrays": [{"method": method, "seed": seed, "split": "val", "bitwise_exact": True,
                          "max_probability_difference": 0.} for method in methods for seed in seeds]}
    reporter.write_json(dataset / "checkpoint_replay.json", replay)
    W, polarity = reporter._base.build_cost_matrix(classes), reporter._base.polarity_vector(classes)
    records = []
    for method in methods:
        for seed in seeds:
            for condition in reporter.CONDITIONS + (reporter.INTERVENTIONS if method == "wavelet_fusion" else ()):
                directory = dataset / f"{method}_seed{seed}"
                filename = "predictions.npz" if condition == "clean" else f"predictions_{condition}.npz"
                path = directory / filename
                if condition != "clean":
                    factor = {"shift_x1": 1, "shift_x4": 2, "awgn_003": 3, "remove_detail": 4, "remove_low": 5}[condition]
                    count = min(len(labels), factor * (1 + seed - min(seeds)))
                    prediction = labels.copy()
                    prediction[:count] = (prediction[:count] + 1) % 8
                    p = np.full((len(labels), 8), .1 / 7)
                    p[np.arange(len(labels)), prediction] = .9
                    np.savez_compressed(path, probabilities=p, labels=labels)
                with np.load(path) as array:
                    p = array["probabilities"].copy()
                metrics = reporter._base._helper.evaluate_predictions(labels, p.argmax(1), W, polarity)
                metrics.update(reporter._base._helper.graph_error_metrics(p, labels, W))
                records.append({"method": method, "seed": seed, "condition": condition,
                                "metrics": metrics, "file": str(path.relative_to(dataset)), "sha256": reporter.file_digest(path)})
    reporter.write_json(dataset / "perturbations.json", records)
    return repo, out, dataset


def test_complete_condition_replay_energy_verification_and_paired_changes(tmp_path):
    repo, _, dataset = fixture(tmp_path)
    summary, audit, rows = reporter.analyze_diagnostics_dataset(dataset, repo)
    assert audit["status"] == "passed"
    assert audit["validation_replay_records_verified"] == 18
    assert audit["test_condition_arrays_sha_and_metrics_verified"] == 78
    assert len(rows) == 78
    change = summary["condition_summary"]["rgb"]["shift_x1"]["change_from_clean"]["acc"]
    assert change["per_seed"] == [-6.25, -12.5, -18.75]
    assert change["mean"] == -12.5
    assert change["sample_sd"] == 6.25
    assert change["negative_pairs"] == 3
    assert not audit["gpu_replay_executed_by_this_report"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "split", "nonexact", "nonzero"])
def test_invalid_replay_coverage_or_exactness_rejected(tmp_path, mutation):
    repo, _, dataset = fixture(tmp_path)
    path = dataset / "checkpoint_replay.json"
    value = reporter.read_json(path)
    if mutation == "missing":
        value["arrays"].pop()
    elif mutation == "duplicate":
        value["arrays"].append(value["arrays"][0])
    elif mutation == "split":
        value["arrays"][0]["split"] = "test"
    elif mutation == "nonexact":
        value["arrays"][0]["bitwise_exact"] = False
    else:
        value["arrays"][0]["max_probability_difference"] = 1e-9
    reporter.write_json(path, value)
    with pytest.raises(ValueError, match="replay"):
        reporter.analyze_diagnostics_dataset(dataset, repo)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "metric", "wrong_path", "sha"])
def test_invalid_condition_coverage_hashes_paths_or_metrics_rejected(tmp_path, mutation):
    repo, _, dataset = fixture(tmp_path)
    path = dataset / "perturbations.json"
    value = reporter.read_json(path)
    if mutation == "missing":
        value.pop()
    elif mutation == "duplicate":
        value.append(value[0])
    elif mutation == "metric":
        value[1]["metrics"]["ecc"] -= .1
    elif mutation == "wrong_path":
        value[1]["file"] = value[0]["file"]
    else:
        value[1]["sha256"] = "0" * 64
    reporter.write_json(path, value)
    with pytest.raises(ValueError, match="coverage|Duplicate|Metric mismatch|different run|hash mismatch"):
        reporter.analyze_diagnostics_dataset(dataset, repo)


@pytest.mark.parametrize("mutation", ["identity", "noise", "noise_expected", "partition", "class_mean", "sample"])
def test_invalid_theory_or_training_sample_diagnostics_rejected(tmp_path, mutation):
    repo, _, dataset = fixture(tmp_path)
    path = dataset / "transform_diagnostics.json"
    value = reporter.read_json(path)
    if mutation == "identity":
        value["maxima"]["parseval_relative_error"] = 1e-5
    elif mutation == "noise":
        value["noise"]["measured_low_fraction"] = .08
    elif mutation == "noise_expected":
        value["noise"]["expected_low_fraction"] = .25
    elif mutation == "partition":
        value["image_energy"][0]["L1_LH_energy_fraction"] += .01
    elif mutation == "class_mean":
        value["class_summary"][0]["normalized_low_energy_fraction"] -= .01
    else:
        value["selected_train_rows"] = list(reversed(value["selected_train_rows"]))
    reporter.write_json(path, value)
    with pytest.raises(ValueError, match="identity|White-noise|partition|summary|sample"):
        reporter.analyze_diagnostics_dataset(dataset, repo)


def test_clean_record_must_match_run_test_result(tmp_path):
    repo, _, dataset = fixture(tmp_path)
    path = dataset / "rgb_seed15/result.json"
    value = reporter.read_json(path)
    value["test"]["ecc"] -= .1
    reporter.write_json(path, value)
    with pytest.raises(ValueError, match="Metric mismatch ecc"):
        reporter.analyze_diagnostics_dataset(dataset, repo)


def test_constant_image_zero_centered_energy_is_supported(tmp_path):
    _, _, dataset = fixture(tmp_path)
    path = dataset / "transform_diagnostics.json"
    value = reporter.read_json(path)
    row = value["image_energy"][0]
    for key in row:
        if key.startswith("centered_") or key == "spatially_centered_low_energy_fraction":
            row[key] = 0.
        elif key == "normalized_low_energy_fraction" or key == "LL2_energy_fraction":
            row[key] = 1.
        elif key.endswith("energy_fraction"):
            row[key] = 0.
    classes = reporter.read_json(dataset / "protocol.json")["classes"]
    value["class_summary"] = _class_summaries(value["image_energy"], classes)
    reporter.write_json(path, value)
    assert reporter.verify_transform_diagnostics(path, np.arange(16) % 8, classes)["status"] == "passed"


def test_standalone_supplement_outputs_and_inference_limits(tmp_path):
    repo, out, _ = fixture(tmp_path)
    summary = reporter.build_diagnostics(out, repo_root=repo)
    assert summary["test_condition_arrays_verified"] == 78
    assert summary["validation_replay_records_verified"] == 18
    for filename in ("diagnostics_summary.json", "diagnostics_verification.json", "condition_results.csv",
                     "wavelet_sensitivity.png", "wavelet_train_energy.png", "DIAGNOSTICS.md", "DIAGNOSTICS.html"):
        assert (out / filename).exists()
    document = (out / "DIAGNOSTICS.html").read_text(encoding="utf8")
    assert "data:image/png;base64," in document
    text = (out / "DIAGNOSTICS.md").read_text(encoding="utf8")
    assert "not a confidence interval" in text
    assert "does not independently rerun network inference" in text
    assert "out-of-distribution" in text
    assert "physical image SNR" in text

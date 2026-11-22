import importlib.util
import json
from pathlib import Path
from argparse import Namespace

import numpy as np
import pytest

from test_seed_report import make_fixture, reporter

SCRIPT = Path(__file__).with_name("report_full_seed_evidence.py")
if not SCRIPT.exists(): SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_full_seed_evidence.py"
spec = importlib.util.spec_from_file_location("full_seed_report", SCRIPT)
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def full_fixture(tmp_path):
    out, repo = make_fixture(tmp_path)
    protocol = reporter.read_json(out / "protocol.json")
    protocol["args"]["methods"] = list(full.METHODS)
    for split, meta in protocol["datasets"].items():
        meta["cache_key"] = "fixture" + split
        cache = repo / "data" / f"{split}_seed_evidence_{meta['cache_key'][:16]}.npy"
        np.save(cache, np.zeros((meta["n"], 3, 192, 192), dtype=np.uint8))
        meta["cache_sha256"] = full.file_hash(cache)
        meta["cache_bytes"] = cache.stat().st_size
    protocol.pop("protocol_hash")
    protocol["protocol_hash"] = reporter.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode())
    reporter.write_json(out / "protocol.json", protocol)
    for method in full.METHODS:
        for seed in protocol["args"]["seeds"]:
            directory = out / f"{method}_seed{seed}"
            result = reporter.read_json(directory / "result.json")
            result["protocol_hash"] = protocol["protocol_hash"]
            reporter.write_json(directory / "result.json", result)
            history = reporter.read_json(directory / "history.json")
            for epoch in history:
                epoch.update(epoch_seconds=2., train_seconds=1.4, optimizer_steps_attempted=2,
                             optimizer_steps_skipped_scale_decrease=0, amp_scale_start=65536, amp_scale_end=65536)
            reporter.write_json(directory / "history.json", history)
    return out, repo, protocol


def test_two_method_shared_analysis_requires_explicit_override(tmp_path):
    out, repo, _ = full_fixture(tmp_path)
    with pytest.raises(ValueError, match="complete study"):
        reporter.analyze_dataset(out, repo, 100, .5)
    ds, verification, _, _ = reporter.analyze_dataset(out, repo, 100, .5, methods=full.METHODS, comparisons=full.COMPARISONS)
    assert set(ds["method_summary"]) == {"ce", "graph"}
    assert set(ds["comparisons"]) == {"graph_minus_ce"}
    assert verification["verified_runs"] == 6
    assert len(ds["diagnostics"]["per_class_seed_metrics"]) == 2 * 3 * 8


def test_cache_hash_tampering_detected(tmp_path):
    out, repo, protocol = full_fixture(tmp_path)
    audit = full.verify_caches(protocol, repo, True)
    assert all(row["sha256_recomputed_and_matched"] for row in audit.values())
    cache = Path(audit["test"]["cache_path"])
    array = np.load(cache)
    array[0, 0, 0, 0] = 1
    np.save(cache, array)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        full.verify_caches(protocol, repo, True)


def test_global_holm_requires_nine_original_tests(tmp_path):
    medium = tmp_path / "medium.json"
    reporter.write_json(medium, {"datasets": {name: {"comparisons": {f"c{i}": {"metrics": {"ecc": {"exact_signflip": {"p_two_sided": .2}}}} for i in range(3)}} for name in ("a", "b", "c")}})
    result = full.global_holm_sensitivity(medium, .01)
    assert result["family_size"] == 10
    assert result["full_comparison_adjusted_p"] == pytest.approx(.1)
    reporter.write_json(medium, {"datasets": {}})
    with pytest.raises(ValueError, match="exactly nine"):
        full.global_holm_sensitivity(medium, .01)


def test_step_counts_and_timing_validation():
    epoch = {"epoch": 1, "epoch_seconds": 2., "train_seconds": 1., "optimizer_steps_attempted": 2,
             "optimizer_steps_skipped_scale_decrease": 1, "amp_scale_start": 65536, "amp_scale_end": 32768}
    result = full.step_diagnostics({("ce", 3): [epoch]}, 16, 8)
    assert result["runs"][0]["inferred_completed"] == 1
    with pytest.raises(ValueError, match="step count"):
        full.step_diagnostics({("ce", 3): [{**epoch, "optimizer_steps_attempted": 3}]}, 16, 8)


def test_full_report_html_fixture_and_strict_design_guard(tmp_path):
    out, repo, _ = full_fixture(tmp_path)
    args = Namespace(out=out, repo=repo, bootstrap=100, meaningful_ecc=.5, verify_cache_hashes=True, medium_summary=None)
    with pytest.raises(ValueError, match="locked"):
        full.report(args)
    full.report(args, strict_design=False)
    verification = reporter.read_json(out / "verification.json")
    assert verification["verified_runs"] == 6
    assert verification["step_diagnostics_checked"]
    assert (out / "epoch_diagnostics.csv").exists()
    assert (out / "seed_results.csv").exists()
    document = (out / "REPORT.html").read_text(encoding="utf8")
    assert "data:image/png;base64," in document
    assert "not a bitwise rerun" in document

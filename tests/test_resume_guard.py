"""Regression tests for experiment identity and epoch-boundary continuation."""
from copy import deepcopy
import json

import pytest
import torch
from torch.nn import functional as F

from ecc.checkpoint import (prepare_run_config, save_checkpoint,
                            select_resume_checkpoint)
from ecc.emotion import WHEEL_ORDER, build_cost_matrix, build_rank_matrix
from ecc.losses import build_loss
from ecc.supervised import scheduled_loss_alpha


def config(output, rule="fixed"):
    return {"seed": 3, "output_dir": str(output),
            "data": {"root": "images", "splits_dir": "splits"},
            "model": {"name": "resnet18", "pretrained": False},
            "loss": {"alpha": 0.0}, "ssl": {"rule": rule},
            "optim": {"epochs": 1, "lr": .01}, "eval": {"select_by": "acc"}}


def checkpoint(path, cfg=None, best=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": 0, "best_epoch": 0, "best_metric": 30.0}
    if cfg is not None:
        record["run_config"] = cfg
    torch.save(record, path)
    if best:
        (path.parent / "best.pt").write_bytes(b"chosen-validation-best")


def test_persistent_checkpoint_is_selected_when_local_missing(tmp_path):
    local, persistent = tmp_path / "local", tmp_path / "persistent"
    checkpoint(persistent / "checkpoint.pt")
    assert select_resume_checkpoint(None, local, persistent) == persistent / "checkpoint.pt"
    checkpoint(local / "checkpoint.pt")
    assert select_resume_checkpoint(None, local, persistent) == local / "checkpoint.pt"
    assert select_resume_checkpoint(persistent / "checkpoint.pt", local, persistent) == persistent / "checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="Explicit resume"):
        select_resume_checkpoint(tmp_path / "missing.pt", local, persistent)


def test_fixed_to_ratio_collision_rejected_before_config_or_best_overwrite(tmp_path):
    out = tmp_path / "completed"
    old = config(out)
    checkpoint(out / "checkpoint.pt", old)
    cfg_file = out / "config.json"
    cfg_file.write_text(json.dumps(old), encoding="utf8")
    original_bytes = cfg_file.read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_run_config(config(out, "ratio"), out)
    assert cfg_file.read_bytes() == original_bytes
    assert (out / "best.pt").read_bytes() == b"chosen-validation-best"


def test_config_only_aborted_setup_allows_corrected_retry(tmp_path):
    out = tmp_path / "aborted"
    out.mkdir()
    old = config(out)
    (out / "config.json").write_text(json.dumps(old), encoding="utf8")
    corrected = config(out)
    corrected["data"]["root"] = "corrected-images"
    selected, provenance = prepare_run_config(corrected, out)
    assert selected is None
    assert provenance["configuration_verification"] == "fresh_run"
    assert json.loads((out / "config.json").read_text())["data"]["root"] == "corrected-images"


def test_epoch_extension_and_runtime_locations_allowed_but_semantics_locked(tmp_path):
    source, out = tmp_path / "source", tmp_path / "new"
    previous = config(source)
    checkpoint(source / "checkpoint.pt", previous)
    current = config(out)
    current["optim"]["epochs"] = 2
    current["resume_from"] = str(source / "checkpoint.pt")
    current["persistent_dir"] = str(tmp_path / "persistent")
    selected, provenance = prepare_run_config(current, out)
    assert selected == source / "checkpoint.pt"
    assert provenance["configuration_verification"] == "embedded_run_config"
    assert provenance["exact_trajectory_resume"] is False
    assert (out / "best.pt").read_bytes() == b"chosen-validation-best"
    for section, key, value in (("loss", "alpha", .3), ("optim", "lr", .02),
                                ("data", "root", "different"), ("model", "name", "resnet50")):
        bad = deepcopy(current)
        bad[section][key] = value
        with pytest.raises(ValueError, match="configuration changed"):
            prepare_run_config(bad, out)


def test_missing_external_best_fails_before_new_config_write(tmp_path):
    source, out = tmp_path / "source", tmp_path / "new"
    checkpoint(source / "checkpoint.pt", config(source), best=False)
    current = config(out)
    current["resume_from"] = str(source / "checkpoint.pt")
    with pytest.raises(FileNotFoundError, match="matching best.pt is missing"):
        prepare_run_config(current, out)
    assert not (out / "config.json").exists()


def test_legacy_checkpoint_without_config_is_explicitly_unverified(tmp_path):
    source, out = tmp_path / "legacy", tmp_path / "new"
    checkpoint(source / "checkpoint.pt")
    cfg = config(out)
    cfg["resume_from"] = str(source / "checkpoint.pt")
    with pytest.warns(RuntimeWarning, match="provenance cannot be verified"):
        _, provenance = prepare_run_config(cfg, out)
    assert provenance["configuration_verification"] == "unverified_legacy_checkpoint"
    assert provenance["upstream_unverified"] is True
    assert json.loads((out / "resume_provenance.json").read_text())["upstream_unverified"] is True


def test_explicit_ce_remains_exact_ce_despite_nonzero_auxiliary_alpha():
    torch.manual_seed(12)
    W = build_cost_matrix(WHEEL_ORDER)
    loss_config = {"kind": "ce", "alpha": 1.0, "switch_epoch": 1}
    criterion = build_loss(loss_config, build_rank_matrix(W))
    logits = torch.randn(5, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 4, 7])
    expected = F.cross_entropy(logits, labels)
    for epoch in (0, 1, 10):
        criterion.alpha = scheduled_loss_alpha(loss_config, epoch)
        assert torch.equal(criterion(logits, labels), expected)
    assert scheduled_loss_alpha({"kind": "listmle", "alpha": .3, "switch_epoch": 1}, 0) == 0
    assert scheduled_loss_alpha({"kind": "listmle", "alpha": .3, "switch_epoch": 1}, 1) == .3


def test_save_checkpoint_embeds_detached_run_config(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    cfg = config(tmp_path)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, 0, 30, 0, run_config=cfg)
    cfg["ssl"]["rule"] = "ratio"
    assert torch.load(path, map_location="cpu")["run_config"]["ssl"]["rule"] == "fixed"

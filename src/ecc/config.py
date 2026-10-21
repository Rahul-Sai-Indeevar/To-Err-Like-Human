from __future__ import annotations

import argparse

import yaml


def load_config(path: str, overrides=None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def parse_args(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override config values, e.g. --set loss.alpha=0.2 optim.epochs=5")
    ap.add_argument("--resume", default=None, metavar="PATH",
                    help="path to a checkpoint.pt to resume training from")
    return ap

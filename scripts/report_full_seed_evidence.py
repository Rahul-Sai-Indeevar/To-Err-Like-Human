"""Full-data paired-seed follow-up; verify results and report one locked contrast.

No GPU computation. Use --verify-cache-hashes to stream input-cache SHA256 checks.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import report_seed_evidence as shared
except ImportError:
    try:
        import report_seed_evidence as shared
    except ImportError:
        from scripts import report_seed_evidence as shared

METHODS = ("ce", "graph")
COMPARISONS = (("graph", "ce"),)
FULL_COUNTS = {"train": 94475, "val": 5904, "test": 17715}


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_caches(protocol, repo, verify_hashes=False):
    root = Path(protocol["args"]["data_root"])
    if not root.is_absolute(): root = repo / root
    audit = {}
    for split in FULL_COUNTS:
        meta = protocol["datasets"][split]
        cache = root / f"{split}_seed_evidence_{meta['cache_key'][:16]}.npy"
        array = np.load(cache, mmap_mode="r", allow_pickle=False)
        if array.shape != (meta["n"], 3, 192, 192) or array.dtype != np.uint8:
            raise ValueError(f"Input cache shape/dtype mismatch: {cache}")
        if cache.stat().st_size != meta["cache_bytes"]:
            raise ValueError(f"Input cache byte count mismatch: {cache}")
        expected = meta["cache_sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid recorded input cache hash")
        if verify_hashes and file_hash(cache) != expected:
            raise ValueError(f"Input cache SHA256 mismatch: {cache}")
        audit[split] = {"cache_path": str(cache.resolve()), "bytes": cache.stat().st_size,
                        "shape": list(array.shape), "dtype": str(array.dtype), "recorded_sha256": expected,
                        "sha256_recomputed_and_matched": bool(verify_hashes)}
        del array
    return audit


def step_diagnostics(histories, train_count, batch_size):
    expected = (train_count + batch_size - 1) // batch_size
    rows = []
    for (method, seed), history in histories.items():
        for epoch in history:
            attempted = epoch["optimizer_steps_attempted"]
            skipped = epoch["optimizer_steps_skipped_scale_decrease"]
            if not isinstance(attempted, int) or attempted != expected or not isinstance(skipped, int) or skipped < 0 or skipped > attempted:
                raise ValueError("Invalid attempted/skipped optimizer-step count")
            seconds, training = epoch["epoch_seconds"], epoch["train_seconds"]
            if not np.isfinite([seconds, training]).all() or training < 0 or seconds < training:
                raise ValueError("Invalid epoch timing")
            if epoch["amp_scale_start"] <= 0 or epoch["amp_scale_end"] <= 0:
                raise ValueError("Invalid AMP scale")
            rows.append({"method": method, "seed": seed, "epoch": epoch["epoch"],
                         "epoch_seconds": seconds, "train_seconds": training,
                         "attempted": attempted, "skipped_scale_decrease": skipped,
                         "amp_scale_start": epoch["amp_scale_start"], "amp_scale_end": epoch["amp_scale_end"]})
    totals = []
    for method, seed in histories:
        matching = [row for row in rows if row["method"] == method and row["seed"] == seed]
        attempted, skipped = sum(row["attempted"] for row in matching), sum(row["skipped_scale_decrease"] for row in matching)
        totals.append({"method": method, "seed": seed, "attempted": attempted,
                       "skipped_scale_decrease": skipped, "inferred_completed": attempted - skipped,
                       "total_epoch_seconds": sum(row["epoch_seconds"] for row in matching)})
    return {"epochs": rows, "runs": totals,
            "definition": "Single-optimizer standard GradScaler scale decreases infer overflow-triggered skipped optimizer steps; saved counts are checked, not independently reconstructed from batch gradients"}


def global_holm_sensitivity(path, full_p):
    summary = shared.read_json(path)
    names, pvalues = [], []
    for name, dataset in summary["datasets"].items():
        for contrast, result in dataset["comparisons"].items():
            names.append(f"{name}/{contrast}")
            pvalues.append(result["metrics"]["ecc"]["exact_signflip"]["p_two_sided"])
    if len(pvalues) != 9:
        raise ValueError("Global sensitivity requires exactly nine declared medium-study ECC comparisons")
    names.append("full_emoset/graph_minus_ce"); pvalues.append(full_p)
    adjusted = shared.holm(pvalues)
    return {"source_summary": str(Path(path).resolve()), "source_summary_sha256": file_hash(path),
            "family_size": len(names), "full_comparison_adjusted_p": adjusted[-1],
            "comparisons": [{"name": n, "raw_p": p, "holm_adjusted_p": a} for n, p, a in zip(names, pvalues, adjusted)],
            "interpretation": "Sensitivity across the earlier nine ECC tests plus this follow-up; not a claim that all ten experiments were planned before any outcomes"}


def plot_results(out, ds, runs, histories, meaningful):
    seeds = ds["seeds"]
    ce = np.array([runs[("ce", seed)]["test"]["ecc"] for seed in seeds])
    graph = np.array([runs[("graph", seed)]["test"]["ecc"] for seed in seeds])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.scatter(ce, graph)
    for x, y, seed in zip(ce, graph, seeds): ax.annotate(str(seed), (x, y), xytext=(4, 4), textcoords="offset points")
    lo, hi = min(ce.min(), graph.min()) - .1, max(ce.max(), graph.max()) + .1
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set(xlabel="CE test ECC (%)", ylabel="Graph test ECC (%)", title="Fresh full-data seed pairs")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax = axes[1]
    ax.scatter(graph - ce, np.arange(len(seeds)))
    ax.axvline(0, linestyle="--", color="gray")
    ax.axvline(meaningful, linestyle=":", color="orange", label=f"+{meaningful:g} practical reference")
    ax.set(yticks=np.arange(len(seeds)), yticklabels=seeds, ylabel="Seed", xlabel="Graph − CE ECC points", title="One independent training pair per dot")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "paired_seeds.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in METHODS:
        curves = np.array([[epoch["val"]["ecc"] for epoch in histories[(method, seed)]] for seed in seeds])
        x, mean = np.arange(1, curves.shape[1] + 1), curves.mean(0)
        sd = curves.std(0, ddof=1)
        line = ax.plot(x, mean, marker="o", label=method)[0]
        ax.fill_between(x, mean - sd, mean + sd, alpha=.15, color=line.get_color())
    ax.set(xlabel="Epoch", ylabel="Validation ECC (%)", title="Full data: mean ± sample training-seed SD")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(out / "validation_curves.png", dpi=180); plt.close(fig)


def report(args, strict_design=True):
    out, repo = args.out.resolve(), args.repo.resolve()
    protocol = shared.read_json(out / "protocol.json")
    if strict_design and (protocol["args"]["seeds"] != list(range(3, 9)) or protocol["args"]["epochs"] != 8 or protocol["args"]["batch_size"] != 64 or {s: protocol["datasets"][s]["n"] for s in FULL_COUNTS} != FULL_COUNTS):
        raise ValueError("Results differ from the locked six-seed/eight-epoch/full-data design")
    ds, verification, runs, histories = shared.analyze_dataset(out, repo, args.bootstrap, args.meaningful_ecc, methods=METHODS, comparisons=COMPARISONS)
    caches = verify_caches(protocol, repo, args.verify_cache_hashes)
    timing = step_diagnostics(histories, ds["split_counts"]["train"], protocol["args"]["batch_size"])
    verification["input_caches"] = caches
    verification["step_diagnostics_checked"] = True
    verification["matches_locked_full_data_design"] = strict_design
    contrast = ds["comparisons"]["graph_minus_ce"]
    ecc = contrast["metrics"]["ecc"]
    p = ecc["exact_signflip"]["p_two_sided"]
    sensitivity = global_holm_sensitivity(args.medium_summary, p) if args.medium_summary is not None else None
    prior = shared.prior_audit(repo).get("full_emoset_8epoch_kaggle")
    summary = {"fresh_full_data": ds, "primary_endpoint": "Fresh-seed mean graph-minus-CE ECC",
               "primary_exact_p": p, "global_holm_sensitivity": sensitivity,
               "prior_three_t4_seeds_not_pooled": prior, "timing_and_amp": timing,
               "primary_scope": "Six seed pairs conditional on the same fixed full-data splits and locked alpha0.3; not independent datasets"}
    shared.write_json(out / "summary.json", summary)
    shared.write_json(out / "verification.json", verification)
    shared.write_json(out / "dataset_provenance.json", ds["data_provenance"])
    shared.write_json(out / "prior_full_data_audit.json", prior)
    with (out / "summary.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "metric", "mean", "sample_seed_sd", "n_seeds"])
        for method, metrics in ds["method_summary"].items():
            for metric, value in metrics.items(): writer.writerow([method, metric, value["mean"], value["sample_sd"], ds["n_seed_pairs"]])
    with (out / "seed_results.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "seed", "best_epoch", *shared.METRICS])
        for (method, seed), result in runs.items():
            writer.writerow([method, seed, result["best_epoch"], *[result["test"][key] for key in shared.METRICS]])
    for filename, rows in (("epoch_diagnostics.csv", timing["epochs"]), ("run_step_totals.csv", timing["runs"])):
        with (out / filename).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot_results(out, ds, runs, histories, args.meaningful_ecc)
    interval, cross = ecc["t95"], contrast["ecc_crossed_bootstrap"]["ci95"]
    description = shared.conclusion(contrast, args.meaningful_ecc)
    lines = ["# Full-data paired-seed replication of the emotion-label graph loss", "",
             description, "",
             "## Locked follow-up design", "",
             f"This is a fresh full-data follow-up commissioned after the smaller-dataset studies. It compares CE with graph alpha=.3, gamma=1, beta=.01, using seeds {ds['seeds']} and {ds['epochs']} epochs per model. It is a new local training study, not restoration of interrupted Kaggle training. Original T4 training already completed its configured eight epochs.", "",
             f"Train/validation/test counts are {ds['split_counts']}; {len(runs)} completed models and {verification['verified_prediction_arrays']} checked prediction arrays. Maximum validation ECC selects each epoch checkpoint; alpha stays locked. All ResNet18 parameters are fine-tuned from ImageNet weights. Initial states and augmentation/order plans match within each seed pair. Both methods receive the same maximum budget of {ds['planned_training_minibatches_per_run']} planned minibatches per model.", "",
             "The graph signal is the eight emotion probabilities, and its inverse-Laplacian residual quadratic emphasizes low label-graph-frequency errors. This is distinct from image spatial frequency. Lower GSEE directly reflects the auxiliary objective and is not independent proof of improved emotion recognition. This focused full-data follow-up has no new isotropic/permutation controls; those mechanisms are examined separately in the medium studies.", "",
             "## Fresh seed results", "",
             "| Method | ACC | ECC | ACC2 | EMC | Brier | GSEE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        metrics = ds["method_summary"][method]
        lines.append("| " + method + " | " + " | ".join(f"{metrics[key]['mean']:.4f} ± {metrics[key]['sample_sd']:.4f}" for key in shared.METRICS) + " |")
    lines += ["", "Means ± sample training-seed SD. ACC/ECC/ACC2/EMC are percentages; Brier/GSEE are raw probability-energy errors.", "",
              f"Graph-minus-CE ECC = {ecc['mean_difference']:+.4f} points; seed Student t95 {interval}; exact two-sided paired sign-flip p={p:.6g}. Positive/negative/zero seed differences: {ecc['positive_pairs']}/{ecc['negative_pairs']}/{ecc['zero_pairs']}. Leave-one-out mean range: {ecc['leave_one_out_mean_range']}.", "",
              f"Crossed seed plus shared class-stratified image bootstrap95: [{cross[0]:+.4f}, {cross[1]:+.4f}], using {args.bootstrap} draws. The same sampled image indices apply across all paired models.", "",
              "![Fresh full-data seed pairs](paired_seeds.png)", "", "![Validation learning curves](validation_curves.png)", "",
              "## Multiplicity and uncertainty", "",
              "The single locked full-data contrast's raw p-value is reported above. The follow-up was added after seeing the medium-study outcomes, so its timing is explicit. A ten-comparison Holm sensitivity can include the earlier nine ECC contrasts without selecting a favorable correction family.", ""]
    if sensitivity is not None:
        lines += [f"Across the earlier nine comparisons plus this one, Holm-adjusted full-data p={sensitivity['full_comparison_adjusted_p']:.6g}. This is a sensitivity analysis, not a claim of a ten-test plan registered before all results. The original medium-only family and raw full-data result remain visible in their respective reports.", ""]
    else:
        lines += ["Ten-comparison sensitivity was not computed because no medium-study summary was supplied. Supply --medium-summary outputs/seed_evidence/summary.json to include it.", ""]
    lines += [f"Student t95 treats seed differences as independent and approximately normally distributed. The exact sign-flip test assumes paired treatment-label exchangeability/sign symmetry under the null. With six pairs its smallest possible two-sided p is 2/64=.03125; it has coarse resolution. The bootstrap cannot create additional independent training seeds. The +{args.meaningful_ecc:g} ECC-point practical reference remains unchanged; nonsignificance is not equivalence, and no TOST or achieved-power claim is made.", "",
              "## Historical T4 comparison, kept separate", ""]
    if prior is not None:
        previous = prior["metrics"]["ecc"]
        lines += [f"The historical three T4 seed pairs 0–2 gave mean graph-minus-CE ECC {previous['mean_difference']:+.4f}, differences {previous['differences']}, seed t95 {previous['t95']} and exact p={previous['exact_signflip']['p_two_sided']:.4g}. The coefficient .3 was chosen using their validation sweep. These historical pairs are not pooled with the fresh six for primary inference.", ""]
    lines += [f"The current protocol records {protocol.get('gpu_name')}, PyTorch {protocol.get('torch')}, torchvision {protocol.get('torchvision')} and deterministic algorithms={protocol.get('deterministic_algorithms')}. The original jobs used T4 workers and a different framework/AMP environment. Split sizes and CSV identities are retained; hardware, runtime and explicit augmentation plans differ. This is evidence about repeatability under the documented local protocol, not a bitwise rerun of the historical training trajectory.", "",
              "## AMP and runtime diagnostics", "",
              "| Method | Seed | Attempted steps | Inferred skipped steps | Inferred completed steps | Epoch time total (s) |", "|---|---:|---:|---:|---:|---:|"]
    for row in timing["runs"]:
        lines.append(f"| {row['method']} | {row['seed']} | {row['attempted']} | {row['skipped_scale_decrease']} | {row['inferred_completed']} | {row['total_epoch_seconds']:.1f} |")
    lines += ["", timing["definition"] + ". Runtime includes validation and is descriptive; no speed advantage is inferred. Epoch-level details are in epoch_diagnostics.csv.", "",
              "## Input provenance and verification", "",
              f"Preparation manifest SHA256: `{ds['data_provenance']['manifest_sha256']}`. Full provenance, pinned revision, source-shard hashes and exclusions are preserved in dataset_provenance.json. The full manifest's exact encoded-byte deduplication does not remove every re-encoded or near-duplicate image.", "",
              "Dataset and CSV hashes, exact label order, finite normalized probabilities, all six reported metrics, every mode energy, roughness, validation checkpoint selection, executed source snapshots and paired plan hashes passed shared verification. Recorded initialization hashes agree; they are not independently regenerated by this report.", "",
              "Cache shapes, dtype and byte counts match the protocol. " + ("All three cache SHA256 values were streamed and independently recomputed by this report." if args.verify_cache_hashes else "Cache SHA256 values are recorded from preparation and were not rehashed by this report; --verify-cache-hashes enables a bounded-memory streaming check."), "",
              "All image-weighted metrics share the original fixed test split. Fresh seeds do not create an independent test dataset or psychological validation. Dataset-specific results and earlier capped-subset outcomes remain separate.", ""]
    text = "\n".join(lines)
    (out / "REPORT.md").write_text(text, encoding="utf8")
    try:
        import markdown
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    except ImportError:
        body = "<pre>" + html.escape(text) + "</pre>"
    for filename in ("paired_seeds.png", "validation_curves.png"):
        uri = "data:image/png;base64," + base64.b64encode((out / filename).read_bytes()).decode()
        if body.startswith("<pre>"): body += f'<img alt="{filename}" src="{uri}">'
        else: body = body.replace(f'src="{filename}"', f'src="{uri}"')
    (out / "REPORT.html").write_text('<!doctype html><html><head><meta charset="utf-8"><title>Full-data paired-seed replication</title><style>body{font:16px/1.6 system-ui;max-width:1150px;margin:40px auto;padding:0 25px}img{max-width:100%}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border:1px solid #ddd}pre{white-space:pre-wrap}</style></head><body>' + body + '</body></html>', encoding="utf8")
    print(json.dumps({"verified_runs": len(runs), "report": str(out / "REPORT.html"), "primary_p": p}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/full_seed_replication"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=1500)
    parser.add_argument("--meaningful-ecc", type=float, default=.5)
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--medium-summary", type=Path)
    args = parser.parse_args()
    if args.bootstrap < 100 or not np.isfinite(args.meaningful_ecc) or args.meaningful_ecc <= 0:
        parser.error("Require bootstrap>=100 and positive finite practical reference")
    report(args)


if __name__ == "__main__":
    main()

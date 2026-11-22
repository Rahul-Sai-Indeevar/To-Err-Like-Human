"""Report recovered full-data alpha sweep and measured spatial-frequency controls."""

import base64, csv, json, argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.metrics import evaluate_predictions, graph_error_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="outputs/full_spatial_frequency")
BASE = Path(parser.parse_args().out)
rows = json.loads((BASE / "results.json").read_text())
audit = json.loads((BASE / "kaggle_audit.json").read_text())
protocol = json.loads((BASE / "protocol.json").read_text())
classes = audit["classes"]
W = build_cost_matrix(classes)
pol = polarity_vector(classes)
assert len(rows) == 96 and protocol["images"] == 17715
conditions = list(dict.fromkeys(r["condition"] for r in rows))
methods = ["ce", "graph_a0p3"]
metrics = ["acc", "acc2", "balanced_acc", "balanced_acc2", "ecc", "emc", "gsee"]
summary = {}
for condition in conditions:
    summary[condition] = {}
    for method in methods:
        rr = [r for r in rows if r["condition"] == condition and r["method"] == method]
        assert sorted(r["seed"] for r in rr) == [0, 1, 2]
        summary[condition][method] = {
            key: {
                "mean": float(np.mean([r["test"][key] for r in rr])),
                "std": float(np.std([r["test"][key] for r in rr], ddof=1)),
            }
            for key in metrics
        }
        summary[condition][method]["clipped_fraction"] = rr[0]["mean_clipped_fraction"]
        summary[condition][method]["mse"] = rr[0]["mean_mse"]
        summary[condition][method]["mean_channel_drift"] = rr[0]["mean_channel_drift"]
# Check every condition's row order and metrics against its saved probabilities.
first = np.load(BASE / "clean/ce_seed0.npz")["labels"]
strata = [np.flatnonzero(first == c) for c in range(8)]
comparisons = {}
for condition in conditions:
    differences = []
    degradation = []
    for seed in [0, 1, 2]:
        c = np.load(BASE / condition / f"ce_seed{seed}.npz")
        g = np.load(BASE / condition / f"graph_a0p3_seed{seed}.npz")
        assert np.array_equal(c["labels"], first) and np.array_equal(g["labels"], first)
        cp = c["probabilities"]
        gp = g["probabilities"]
        assert cp.shape == gp.shape == (17715, 8)
        assert np.isfinite(cp).all() and np.isfinite(gp).all()
        assert np.allclose(cp.sum(1), 1, atol=1e-6) and np.allclose(
            gp.sum(1), 1, atol=1e-6
        )
        for name, probs in [("ce", cp), ("graph_a0p3", gp)]:
            check = evaluate_predictions(first, probs.argmax(1), W, pol)
            check.update(graph_error_metrics(probs, first, W))
            saved = next(
                r["test"]
                for r in rows
                if r["condition"] == condition
                and r["method"] == name
                and r["seed"] == seed
            )
            for key in ["acc", "ecc", "emc", "acc2", "gsee"]:
                assert np.isclose(check[key], saved[key], atol=1e-8)
        d = 100 * (1 / W[first, gp.argmax(1)] - 1 / W[first, cp.argmax(1)])
        baseline_c = np.load(BASE / "clean" / f"ce_seed{seed}.npz")[
            "probabilities"
        ].argmax(1)
        baseline_g = np.load(BASE / "clean" / f"graph_a0p3_seed{seed}.npz")[
            "probabilities"
        ].argmax(1)
        clean_d = 100 * (1 / W[first, baseline_g] - 1 / W[first, baseline_c])
        differences.append(d)
        degradation.append(d - clean_d)
    d = np.mean(differences, axis=0)
    dd = np.mean(degradation, axis=0)
    rng = np.random.default_rng(20261115)
    draws = []
    degradation_draws = []
    for _ in range(1000):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        draws.append(float(d[ix].mean()))
        degradation_draws.append(float(dd[ix].mean()))
    comparisons[condition] = {
        "ecc_graph_minus_ce": float(d.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "ecc_advantage_minus_clean_advantage": float(dd.mean()),
        "degradation_ci95": np.quantile(degradation_draws, [0.025, 0.975]).tolist(),
    }
(BASE / "summary.json").write_text(json.dumps(summary, indent=2))
(BASE / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2))
with (BASE / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["condition", "method"]
        + [k for m in metrics for k in [m + "_mean", m + "_std"]]
        + ["clipped_fraction", "mse", "mean_channel_drift"],
    )
    writer.writeheader()
    for condition in conditions:
        for method in methods:
            z = summary[condition][method]
            writer.writerow(
                dict(
                    condition=condition,
                    method=method,
                    **{
                        m + "_" + stat: z[m][stat]
                        for m in metrics
                        for stat in ["mean", "std"]
                    },
                    **{
                        k: z[k]
                        for k in ["clipped_fraction", "mse", "mean_channel_drift"]
                    },
                )
            )
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, mode in enumerate(["lowpass", "highpass_dc"]):
    for j, metric in enumerate(["acc", "acc2", "ecc"]):
        ax = axes[i, j]
        for method in methods:
            rr = sorted(
                [r for r in rows if r["mode"] == mode and r["method"] == method],
                key=lambda r: r["cutoff"],
            )
            cutoffs = sorted(set(r["cutoff"] for r in rr))
            names = [f"{mode}_{c:g}" for c in cutoffs]
            mean = [summary[n][method][metric]["mean"] for n in names]
            sd = [summary[n][method][metric]["std"] for n in names]
            ax.errorbar(cutoffs, mean, yerr=sd, marker="o", capsize=3, label=method)
            ax.axhline(summary["clean"][method][metric]["mean"], ls=":", alpha=0.4)
        ax.set(
            xlabel="Cutoff (cycles/pixel)", ylabel=metric.upper() + " (%)", title=mode
        )
        ax.set_xscale("log")
        ax.set_xticks(cutoffs, [f"{c:g}" for c in cutoffs], rotation=25)
        ax.minorticks_off()
        ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(BASE / "frequency_response.png")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, mode in zip(axes, ["lowpass", "highpass_dc"]):
    cutoffs = protocol["cutoffs_cycles_per_pixel"]
    for method in methods:
        for metric, chance, label in [
            ("balanced_acc", 12.5, "eight-class"),
            ("balanced_acc2", 50.0, "polarity"),
        ]:
            clean = summary["clean"][method][metric]["mean"]
            retention = [
                (summary[f"{mode}_{c:g}"][method][metric]["mean"] - chance)
                / (clean - chance)
                for c in cutoffs
            ]
            ax.plot(cutoffs, retention, "o-", label=method + " " + label)
    ax.axhline(0, color="black", lw=0.7)
    ax.axhline(1, color="gray", ls=":")
    ax.set(
        xlabel="Cutoff (cycles/pixel)",
        ylabel="Chance-adjusted clean performance retained",
        title=mode,
    )
    ax.set_xscale("log")
    ax.set_xticks(cutoffs, [f"{c:g}" for c in cutoffs], rotation=25)
    ax.minorticks_off()
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(BASE / "chance_adjusted_retention.png")
plt.close(fig)
fig, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(len(conditions))
p = [comparisons[c]["ecc_graph_minus_ce"] for c in conditions]
low = [comparisons[c]["ci95"][0] for c in conditions]
high = [comparisons[c]["ci95"][1] for c in conditions]
ax.errorbar(x, p, yerr=[np.array(p) - low, np.array(high) - p], fmt="o", capsize=3)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, conditions, rotation=60, ha="right")
ax.set(
    ylabel="Graph minus CE ECC (percentage points)",
    title="Paired image-bootstrap intervals; exploratory, no multiplicity correction",
)
fig.tight_layout()
fig.savefig(BASE / "paired_ecc.png")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
names = ["clean", "phase_zero", "phase_random", "amplitude_flat"]
x = np.arange(4)
for ax, metric in zip(axes, ["acc", "ecc"]):
    for j, method in enumerate(methods):
        ax.bar(
            x + (j - 0.5) * 0.35,
            [summary[n][method][metric]["mean"] for n in names],
            0.35,
            yerr=[summary[n][method][metric]["std"] for n in names],
            label=method,
            capsize=3,
        )
    ax.set_xticks(x, names, rotation=20)
    ax.set_ylabel(metric.upper() + " (%)")
    ax.legend()
fig.tight_layout()
fig.savefig(BASE / "phase_controls.png")
plt.close(fig)
lines = [
    "# Full EmoSet recovery and spatial-frequency follow-up",
    "",
    "## What happened in Kaggle",
    "",
    "Training did not stop early. All 15 runs finished their configured eight epochs, with complete histories, best-validation checkpoints and full-test predictions. The original notebook filename mentions 25 epochs, but its executed MAX_EPOCHS, MIN_EPOCHS, checkpoint metadata and histories all specify eight. The failure occurred after report generation and ZIP creation: archive.stat().st_size() calls an integer property. The correct expression is archive.stat().st_size. A second display cell repeats the same bug. Both calls are corrected in kaggle/sp_full_alpha_sweep_fixed.ipynb and kaggle/alpha_sweep_dual_t4_runner.py. No original Downloads file was overwritten.",
    "",
    "All 15 sets of 17,715 saved test labels/probabilities and confusion matrices were checked. ACC, ECC, EMC, ACC2, GSEE and Brier were recomputed, and selected epochs were checked against validation ECC. The alpha choice was independently recovered from validation scores. This is result-artifact verification, not a rerun of Kaggle training.",
    "",
    "## Recovered full-data results",
    "",
    "The dataset contains 94,475 train, 5,904 validation and 17,715 test images after exact encoded-byte duplicate removal. This does not detect re-encoded duplicates or establish psychological label validity. Means below use three training seeds; +/- values are sample seed SD.",
    "",
    "| Alpha | ACC (%) | ECC (%) | EMC (%) | Mean best validation ECC |",
    "|---|---:|---:|---:|---:|",
]
for method in ["ce", "graph_a0p1", "graph_a0p3", "graph_a0p5", "graph_a1p0"]:
    z = audit["summary"][method]
    alpha = (
        "0 (CE)" if method == "ce" else method.removeprefix("graph_a").replace("p", ".")
    )
    lines.append(
        "| "
        + alpha
        + " | "
        + " | ".join(
            f"{z[m]['mean']:.3f} +/- {z[m]['std']:.3f}" for m in ["acc", "ecc", "emc"]
        )
        + f" | {z['validation_mean']:.3f} |"
    )
lines += [
    "",
    "Validation selects alpha=0.3. Its full-test improvement is +0.359 ACC points and +0.268 ECC points over CE. The supplied paired class-stratified 95% image-bootstrap intervals were [0.091,0.653] for ACC and [0.078,0.471] for ECC. They are conditional on these three fitted models, not confidence over future training seeds. Alpha=1 has a slightly higher test score, but replacing the validation-selected alpha with it would use test data for model selection.",
    "",
    "## Next experiment actually executed",
    "",
    f"Six frozen checkpoints (CE and graph alpha=0.3, seeds 0/1/2) were evaluated on all {protocol['images']:,} test images in 16 conditions: clean, six low-pass cutoffs, six high-pass-plus-DC cutoffs, zero phase, randomized phase, and flattened non-DC amplitude. This is 96 model-condition evaluations / {96 * protocol['images']:,} image-model predictions. A 1,600-image execution pilot preceded this full evaluation; the same transform conditions were retained. There was no retraining or new alpha selection.",
    "",
    "All conditions act on the same center-cropped 160x160 RGB inputs and the same images across all models. Cutoffs are 0.025, 0.05, 0.1, 0.2, 0.35 and 0.5 cycles/pixel. Signals remain in 0..255 scale before the saved model normalization. No per-image min-max rescaling is applied.",
    "",
    "## Signal-processing theory and controls",
    "",
    "Image FFT frequency is spatial variation in cycles/pixel. It is different from the label-graph eigenvalues in the GFT loss. The orthonormal two-dimensional FFT obeys Parseval energy preservation and separates a complex spectrum into magnitude and phase. Each RGB channel is transformed independently.",
    "",
    "For radial spatial frequency r, the image-processing Butterworth-style mask is H_L(r)=1/[1+(r/f_c)^(2n)] with n=2. This explicit definition has half-amplitude response at the stated cutoff (it is not a claim of the continuous-time prototype's -3 dB amplitude convention). High-pass is 1-H_L, with the DC bin reset to one. Keeping DC avoids confounding high-pass removal with an almost-black image and preserves each channel's mean before clipping. Frequencies are periodic on the finite image grid; boundary discontinuities can influence the response.",
    "",
    "Zero-phase and randomized-phase controls preserve Fourier magnitudes before clipping. Flattened amplitude preserves original phase, DC and pre-clipping AC energy. Real-valued reconstruction follows from Hermitian symmetry. Tests check magnitude preservation, Parseval, DC/mean preservation and the complementary-filter identity. These conditions are severe distribution shifts, not proofs of human perceptual mechanisms.",
    "",
    "All reconstructions are clipped to [0,255] for classifier input. Clipping is nonlinear: it changes Fourier magnitudes and can alter means. The clipping fraction, resulting MSE and per-channel mean drift are saved for every condition. Consequently, phase results cannot be attributed to a pure phase intervention after clipping. In particular, high post-clipping damage makes an amplitude-preserved claim valid only before clipping.",
    "",
    "Exact emotion has eight classes and polarity has two. Their raw accuracies have different chance levels. The retention plot uses balanced accuracies and (filtered-chance)/(clean-chance), with 12.5% chance for eight classes and 50% for polarity, to avoid interpreting the easier binary task as automatic evidence of coarse-to-fine perception.",
    "",
    "![Frequency-response curves](frequency_response.png)",
    "",
    "![Chance-adjusted retention](chance_adjusted_retention.png)",
    "",
    "## Measured condition results",
    "",
    "| Condition | CE ECC | Graph ECC | Graph-CE ECC points | Conditional 95% interval | Pixel clipping |",
    "|---|---:|---:|---:|---|---:|",
]
for c in conditions:
    a = summary[c]["ce"]
    b = summary[c]["graph_a0p3"]
    d = comparisons[c]
    lines.append(
        f"| {c} | {a['ecc']['mean']:.3f} | {b['ecc']['mean']:.3f} | {d['ecc_graph_minus_ce']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {100 * a['clipped_fraction']:.2f}% |"
    )
lines += [
    "",
    "![Paired ECC comparisons](paired_ecc.png)",
    "",
    "Intervals use 1,000 paired class-stratified image bootstrap replicates, with the same resampled indices shared across all model seeds. They preserve the observed class counts and are conditional on these models. There is no multiplicity correction across the 16 inspected conditions; treat the results as exploratory. Degradation-adjusted graph advantages (filtered advantage minus clean advantage) are separately saved in paired_comparisons.json to distinguish robustness from a pre-existing clean-score difference.",
    "",
    "## Findings supported by these measurements",
    "",
]
worst = min(
    [c for c in conditions if c.startswith("lowpass")],
    key=lambda c: comparisons[c]["ecc_graph_minus_ce"],
)
best = max(
    [c for c in conditions if c.startswith("highpass")],
    key=lambda c: comparisons[c]["ecc_graph_minus_ce"],
)
lines += [
    f"The graph objective is not uniformly more robust to spatial filtering. At {worst}, graph-minus-CE ECC is {comparisons[worst]['ecc_graph_minus_ce']:+.3f} points; at {best} it is {comparisons[best]['ecc_graph_minus_ce']:+.3f} points. These condition-dependent outcomes show that label-graph regularization and image-frequency robustness are distinct questions.",
    "",
    f"Clean replay on this machine gives CE ECC {summary['clean']['ce']['ecc']['mean']:.3f} and graph ECC {summary['clean']['graph_a0p3']['ecc']['mean']:.3f}. Small differences from the saved Kaggle clean scores are measured in checkpoint_replay.json: T4 versus RTX 5050 and framework/AMP differences mean replay is numerically close rather than bitwise identical.",
    "",
    "Phase and amplitude manipulations reduce recognition substantially compared with clean images. That supports the dependence of these fitted classifiers on the altered signal structure. It does not establish that phase exclusively encodes emotion, that low frequencies always preserve polarity, or that this experiment proves Reverse Hierarchy Theory. The chance-adjusted retention curves and clipping diagnostics are necessary to assess those narrower hypotheses.",
    "",
    "![Phase/amplitude controls](phase_controls.png)",
    "",
    "## Continuation and reproducibility",
    "",
    "The next experiment is complete locally and its probabilities/labels are saved under outputs/full_spatial_frequency. A future 25-epoch study would be a new training-budget comparison, not recovery of missing epochs. Preserve equal budgets for CE and a graph coefficient locked using validation; do not choose a new alpha from this test study. The original last_resume.pt files include model, optimizer and AMP scaler but omit RNG states, so they cannot guarantee an exact uninterrupted training trajectory. They remain in the original workspace ZIP.",
    "",
    "Commands (activate the project environment or use its Python executable):",
    "",
    "```powershell",
    'python scripts/import_sp_kaggle.py "C:/Users/thota/Downloads/_output_.zip"',
    "python scripts/run_spatial_frequency.py --per-class 0 --out outputs/frequency_reproduction",
    "python scripts/report_spatial_frequency.py --out outputs/frequency_reproduction",
    "```",
    "",
    "The local CUDA runtime lives in work/cuda_runtime; use scripts/run_spatial_frequency.ps1 on this machine. For another machine install compatible CUDA PyTorch normally. Protocol and source snapshots are saved with the experiment; use a fresh output directory if protocol settings change. No repository push was performed.",
    "",
    "Primary references: [original emotion evaluation paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2bb120e9a2cb9c2a50921b7f865c421-Abstract-Conference.html), [graph signal processing](https://arxiv.org/abs/1211.0053), [2D Fourier image processing](https://docs.opencv.org/4.13.0/de/dbc/tutorial_py_fourier_transform.html).",
]
(BASE / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
body = markdown.markdown("\n".join(lines), extensions=["tables", "fenced_code"])
for p in BASE.glob("*.png"):
    body = body.replace(
        f'src="{p.name}"',
        f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"',
    )
css = "body{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;color:#152839}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow:auto}th,td{padding:8px;border:1px solid #cad5df}th{background:#edf4fa}img{max-width:100%}pre{overflow:auto;background:#f1f4f7;padding:15px}"
(BASE / "REPORT.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>Full EmoSet recovery and FFT experiments</title><style>'
    + css
    + "</style>"
    + body,
    encoding="utf8",
)
print(BASE / "REPORT.html")

selection = json.loads((BASE / "selected_images.json").read_text())
assert [r["test_index"] for r in selection] == list(range(17715))
verification = {
    "verified_original_runs": 15,
    "verified_epochs_each": 8,
    "verified_frequency_evaluations": 96,
    "test_images_per_evaluation": 17715,
    "image_model_predictions": 17715 * 96,
    "metric_recomputation": "ACC/ECC/EMC/ACC2/GSEE recomputed from each of 96 saved probability arrays",
    "actual_sampling": "all original test rows, in source order",
    "protocol_description_note": "The executed runner's generic sample text said balanced subset; per_class=0 and selected row IDs verify that the actual evaluation used the entire test split.",
    "checkpoint_replay": json.loads((BASE / "checkpoint_replay.json").read_text()),
}
(BASE / "verification.json").write_text(json.dumps(verification, indent=2))

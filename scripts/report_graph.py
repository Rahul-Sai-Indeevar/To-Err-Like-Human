"""Generate plots, paired uncertainty, and a theory/results report from saved runs."""

import json
import base64
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecc.emotion import build_cost_matrix, polarity_vector
from ecc.graph import graph_basis, spectral_weights
from ecc.metrics import evaluate_predictions

BASE = Path("outputs/graph_experiment")
BASE.mkdir(parents=True, exist_ok=True)
classes = Path("data/emoset_graph/classes.txt").read_text().splitlines()
W = build_cost_matrix(classes)
A, L, v, U = graph_basis(W)
polarity = polarity_vector(classes)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].imshow(A, cmap="viridis")
axes[0].set_title("Emotion graph adjacency: A = 1/W, diagonal 0")
axes[0].set_xticks(range(8), classes, rotation=80)
axes[0].set_yticks(range(8), classes)
axes[1].plot(range(8), v, "o-")
axes[1].set(
    xlabel="Graph mode k",
    ylabel="Laplacian eigenvalue",
    title="Graph frequency spectrum",
)
axes[2].barh(classes, U[:, 1])
axes[2].axvline(0, color="black")
axes[2].set_title("Fiedler vector (sign is arbitrary)")
fig.tight_layout()
fig.savefig(BASE / "graph_theory.png")
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4))
for tau in [0, 0.25, 1, 3]:
    ax.plot(v, np.exp(-tau * v), "o-", label=f"t={tau}")
ax.set(
    xlabel="Graph frequency (eigenvalue)",
    ylabel="Heat-filter gain",
    title="True graph low-pass response exp(-t lambda)",
)
ax.legend()
fig.tight_layout()
fig.savefig(BASE / "heat_filter.png")
plt.close(fig)
theory = {
    "classes": classes,
    "W": W.tolist(),
    "A": A.tolist(),
    "L": L.tolist(),
    "eigenvalues": v.tolist(),
    "eigenvectors": U.tolist(),
    "fiedler_polarity_abs_correlation": float(
        abs(np.corrcoef(U[:, 1], polarity)[0, 1])
    ),
    "fiedler_sign_separates_polarity": bool(
        np.all((U[:, 1] > 0) == polarity) or np.all((U[:, 1] < 0) == polarity)
    ),
    "orthogonality_error": float(np.linalg.norm(U.T @ U - np.eye(8))),
    "laplacian_reconstruction_error": float(np.linalg.norm(L - U @ np.diag(v) @ U.T)),
}
(BASE / "graph_theory.json").write_text(json.dumps(theory, indent=2))
fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(U.T, cmap="coolwarm", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(8), classes, rotation=35, ha="right")
ax.set_yticks(range(8), [f"k={k}, lambda={v[k]:.3f}" for k in range(8)])
ax.set_title("All eight graph Fourier basis modes (signs arbitrary)")
fig.colorbar(im, ax=ax, label="Eigenvector coefficient")
fig.tight_layout()
fig.savefig(BASE / "gft_basis.png")
plt.close(fig)
manifest = json.loads(Path("data/emoset_graph/manifest.json").read_text())
lines = [
    "# Graph signal processing for visual emotion recognition",
    "",
    "## Scope and provenance",
    "",
    "This experiment implements the attached Idea 1 on real EmoSet photographs. The eight class probabilities form a signal on the LABEL graph. Graph frequency is variation across emotion vertices; it is not spatial frequency in the input photograph. Idea 2 is deferred.",
    "",
    "The source repository was audited at base commit 21abe4fa8ea36fe23139e6c18a41d8491bcaffef. It already contained ResNet training, ACC/ACC2/ECC/EMC, ListMLE, FixMatch threshold adaptation, and minimum-cost decoding. README results are historical reported results, not independently rerun FI benchmarks. GSP was absent. The official author code is partial; its metric.py does not define the wheel. This implementation preserves the existing wheel and checks invariance rather than silently redefining it. S2-VER integration and user study remain outside this experiment.",
    "",
    f"Mirror: {manifest['source']['repository']} at revision {manifest['source']['revision']}. Source shard SHA256 hashes and exact selected image identifiers are in data/emoset_source/provenance.json and data/emoset_graph/*csv. The subset preserves the mirror train/val/test assignments. It is a deterministic capped sample from selected shards, not a random sample of all 118K images, and is not the official full benchmark.",
    "",
    "| Split | Images | Per-class counts |",
    "|---|---:|---|",
]
for s in ["train", "val", "test"]:
    lines.append(f"| {s} | {manifest[s]['selected']} | {manifest[s]['per_class']} |")
lines += [
    "",
    "RGB pixel hashes remove exact duplicates within and across splits; invalid images are excluded and logged. Near-duplicates are not detected. Images are stored at at most 320 pixels on their longest side. Model transforms resize to 192x192 and crop to 160x160. Train crops/flips are used for full fine-tuning. These choices differ from the paper and limit numerical comparability.",
    "",
    "## Mathematical connection to signal processing",
    "",
    "Let A_ij = 1/W_ij for i != j, A_ii = 0, D_ii = sum_j A_ij, and L = D-A = U diag(lambda) U^T. This is a dense weighted graph induced by the wheel distances, not a ring-only adjacency. The GFT of a class-probability signal p is U^T p; its inverse is U p_hat. U is orthonormal, giving Parseval energy preservation.",
    "",
    "The Dirichlet identity is p^T L p = 1/2 sum_ij A_ij (p_i-p_j)^2 = sum_k lambda_k p_hat_k^2. Thus high graph frequency means rapid variation across strongly connected emotion vertices. No physiological time or physical hertz is involved.",
    "",
    f"Computed eigenvalues: {np.round(v, 6).tolist()}. The Fiedler vector separates polarity: {theory['fiedler_sign_separates_polarity']}; absolute correlation with the polarity vector: {theory['fiedler_polarity_abs_correlation']:.6f}. This follows from the chosen label geometry and is not a learned discovery about human brains. Eigenvector signs are arbitrary.",
    "",
    "Implemented objective: CE + alpha [sum_(k>0) lambda_k^(-gamma) |u_k^T(p-y)|^2 + beta p^T L p]. Default alpha=1, gamma=1, beta=0.01. The spectral residual term weights LOW-frequency errors more heavily. For gamma=1 it equals (p-y)^T L^+ (p-y), using the Laplacian pseudoinverse. It is an inverse-order graph norm, not the usual positive-order Sobolev/Dirichlet penalty. It is not itself a low-pass filter applied to p. A true heat filter has response exp(-t lambda); its response is plotted separately. At gamma=0 the spectral residual term equals the Brier error because the residual DC component vanishes.",
    "",
    "Both p and one-hot y sum to one, so their residual has zero DC. Excluding the zero eigenvalue is mathematically necessary. The Dirichlet penalty alone favors a constant (uniform) probability vector; CE is required to preserve class discrimination. Smaller energy is not automatically higher accuracy. The supplied text's claims that ListMLE is computationally heavy, that EMC is an SNR, or that weighted confusion scoring is a physical matched filter are not established by these experiments.",
    "",
    "GSEE is defined here as mean inverse-eigenvalue-weighted probability residual energy. It is a descriptive experimental quantity, not a validated replacement for ECC/EMC. Per-mode mean squared residuals are graph error energy, not an estimated stationary PSD. Repeated eigenvalues must be interpreted using total eigenspace energy rather than individual basis vectors.",
    "",
    "## Protocol and comparisons",
    "",
    "All methods use the same class order, data partitions, three seeds (0,1,2), pretrained ImageNet ResNet18 weights, optimizer within a stage, and maximum validation ECC checkpoint selection. Test data never chooses epochs or hyperparameters. Coefficients are fixed in advance; there is no post-test tuning. The frozen-feature probe isolates loss effects at low computational cost and includes no-smooth, flat-frequency and uniform-graph controls. Full-backbone fine-tuning separately compares CE, CE+0.2 ListMLE, and CE+graph. The two stages must not be combined into a single ranking.",
    "",
    "ACC/ECC/EMC/ACC2 are percentages, higher better. Brier and GSEE are raw probability-energy errors, lower better. EMC describes a changing subset of mistakes, so error counts accompany it. Uniform-graph performance is scored on the same emotional graph as all other methods. Seed standard deviations below are sample standard deviations.",
]
allsummary = {}
for stage in ["probe", "finetune"]:
    results = [
        json.loads(p.read_text()) for p in sorted((BASE / stage).glob("*/result.json"))
    ]
    if not results:
        continue
    methods = list(dict.fromkeys(r["method"] for r in results))
    summary = {}
    lines += [
        "",
        f"## Executed {stage} results",
        "",
        "| Method | Runs | ACC | ECC | EMC | ACC2 | GSEE | Seconds/run |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in methods:
        rs = [r for r in results if r["method"] == m]
        vals = {}
        for key in ["acc", "ecc", "emc", "acc2", "gsee", "brier", "n_errors"]:
            a = np.array([r["test"][key] for r in rs])
            vals[key] = {
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            }
        summary[m] = vals
        fmt = lambda k: f"{vals[k]['mean']:.3f} +/- {vals[k]['std']:.3f}"
        lines.append(
            "| "
            + m
            + f" | {len(rs)} | "
            + " | ".join(fmt(k) for k in ["acc", "ecc", "emc", "acc2", "gsee"])
            + f" | {np.mean([r['seconds'] for r in rs]):.1f} |"
        )
    lines += [
        "",
        "Heat diffusion is selected only using validation ECC from t in {0, 0.1, 0.3, 1.0}. The following held-out results are separate from the primary unfiltered scores:",
        "",
        "| Method | Mean selected t | Filtered ECC | Filtered ACC |",
        "|---|---:|---:|---:|",
    ]
    for m in methods:
        rs = [r for r in results if r["method"] == m and "heat_filter" in r]
        if rs:
            lines.append(
                f"| {m} | {np.mean([r['heat_filter']['chosen_time'] for r in rs]):.3f} | {np.mean([r['heat_filter']['test']['ecc'] for r in rs]):.3f} | {np.mean([r['heat_filter']['test']['acc'] for r in rs]):.3f} |"
            )
    allsummary[stage] = summary
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(methods))
    axes[0].bar(
        x,
        [summary[m]["ecc"]["mean"] for m in methods],
        yerr=[summary[m]["ecc"]["std"] for m in methods],
        capsize=4,
    )
    axes[0].set_xticks(x, methods, rotation=30, ha="right")
    axes[0].set(ylabel="ECC (%)", title=stage + " held-out ECC (mean +/- seed SD)")
    for m in methods:
        energy = np.mean(
            [r["test"]["error_energy"] for r in results if r["method"] == m], axis=0
        )
        axes[1].plot(v[1:], energy[1:], "o-", label=m)
    axes[1].set(
        xlabel="Graph frequency (lambda)",
        ylabel="Mean squared GFT residual",
        title="Graph error energy (DC excluded)",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE / f"{stage}_comparison.png")
    plt.close(fig)
    lines += ["", f"![{stage} results]({stage}_comparison.png)", ""]
    if stage == "finetune":
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for method in methods:
            histories = []
            for path in sorted((BASE / stage).glob(f"{method}_seed*/history.json")):
                histories.append(
                    [item["val"]["ecc"] for item in json.loads(path.read_text())]
                )
            if histories:
                h = np.asarray(histories)
                mean = h.mean(0)
                sd = h.std(0, ddof=1) if len(h) > 1 else np.zeros(h.shape[1])
                epochs = np.arange(1, h.shape[1] + 1)
                ax.plot(epochs, mean, "o-", label=method)
                ax.fill_between(epochs, mean - sd, mean + sd, alpha=0.15)
        ax.set(
            xlabel="Training epoch",
            ylabel="Validation ECC (%)",
            title="Full-backbone learning curves (mean +/- seed SD)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(BASE / "learning_curves.png")
        plt.close(fig)
        lines += ["![Validation learning curves](learning_curves.png)", ""]
        fig, axes = plt.subplots(
            1, len(methods), figsize=(5 * len(methods), 4.8), squeeze=False
        )
        for ax, method in zip(axes[0], methods):
            matrices = [
                np.loadtxt(p, delimiter=",")
                for p in sorted((BASE / stage).glob(f"{method}_seed*/confusion.csv"))
            ]
            S = np.sum(matrices, axis=0)
            normalized = 100 * S / S.sum(axis=1, keepdims=True)
            ax.imshow(normalized, vmin=0, vmax=100, cmap="Blues")
            ax.set_xticks(range(8), classes, rotation=70)
            ax.set_yticks(range(8), classes)
            ax.set(
                title=method + " confusion (%)", xlabel="Predicted", ylabel="Reference"
            )
            for i in range(8):
                for j in range(8):
                    ax.text(
                        j,
                        i,
                        f"{normalized[i, j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if normalized[i, j] > 55 else "black",
                    )
        fig.tight_layout()
        fig.savefig(BASE / "confusion_matrices.png")
        plt.close(fig)
        lines += [
            "![Aggregated held-out confusion matrices](confusion_matrices.png)",
            "",
        ]
    if "graph" in summary and "ce" in summary:
        rng = np.random.default_rng(2026)
        diffs = []
        pairs = []
        for seed in [0, 1, 2]:
            ca = BASE / stage / f"ce_seed{seed}" / "predictions.npz"
            ga = BASE / stage / f"graph_seed{seed}" / "predictions.npz"
            if ca.exists() and ga.exists():
                c = np.load(ca)
                g = np.load(ga)
                assert np.array_equal(c["labels"], g["labels"])
                y = c["labels"]
                pc = c["probabilities"].argmax(1)
                pg = g["probabilities"].argmax(1)
                pairs.append(100 * (1 / W[y, pg] - 1 / W[y, pc]))
        if pairs:
            d = np.mean(pairs, axis=0)
            for _ in range(2000):
                diffs.append(float(d[rng.integers(0, len(d), len(d))].mean()))
            low, high = np.quantile(diffs, [0.025, 0.975])
            delta = float(d.mean())
            lines += [
                f"Graph minus CE ECC: {delta:.3f} percentage points. Paired image-bootstrap 95% interval: [{low:.3f}, {high:.3f}]. The same resampled image indices are shared across seeds. This interval is conditional on these trained models and this shard subset; it does not estimate generalization across all possible training seeds or populations.",
                "",
            ]
            if low > 0:
                verdict = "This subset supports an ECC benefit conditional on the tested models, but broader validation is needed."
            elif high < 0:
                verdict = "The proposed graph objective underperformed CE on this subset; the experiment does not support an improvement claim."
            else:
                verdict = "The paired interval includes zero; this experiment does not establish an ECC improvement."
            lines += [verdict, ""]
lines += [
    "## Electrical-network interpretation of the spectral residual",
    "",
    "For gamma=1, the residual z=p-y has zero total mass and the graph loss is z^T L^+ z. Let a zero-mean potential q solve Lq=z. Then z^T L^+ z=q^T Lq, the same Dirichlet energy identity as a balanced-current resistor network whose conductances are A. For a hard wrong prediction j on reference i, z=e_j-e_i and the quantity becomes the effective resistance between those vertices. Thus the proposed loss has an exact network-energy interpretation. The actual graph is a mathematical label-similarity model, not a measured physical circuit or a measurement of human emotions.",
    "",
    "![All GFT modes](gft_basis.png)",
    "",
    "## What is established and what remains open",
    "",
    "The graph construction, transform inversion, Parseval identity, Dirichlet identity, pseudoinverse interpretation, permutation invariance and differentiable optimization have explicit tests. A loss improvement is a separate empirical question. Three seeds and a capped shard subset provide a controlled pilot rather than a full NeurIPS reproduction. Data-label subjectivity, shard-selection bias, shorter training, fixed loss scaling, and ImageNet transfer limit conclusions. An inverse-frequency loss can improve probability energy without changing argmax decisions; that is not a contradiction.",
    "",
    "The graph was constructed from the same wheel used by ECC/EMC. Good agreement between GSEE and ECC is not independent psychological validation. Checkpoint selection on ECC favors that target; ACC is reported alongside it. The supplied proposal's expectation that all metrics improve is not a required or guaranteed outcome.",
    "",
    "Uniform-graph controls remove emotion geometry. The raw uniform graph also changes the eigenvalue scale; the additional trace-matched uniform control preserves total edge weight to reduce that confound. It does not match every eigenvalue or inverse-spectrum moment. gamma=0 changes both weighting shape and auxiliary-loss scale. Interpret these ablations as diagnostic comparisons, not perfect causal isolation. Runtime measurements include validation and overlapping CPU/GPU studies and do not establish an isolated speed advantage.",
    "",
    "The theoretical smoothing failure is explicit: uniform probabilities have zero Dirichlet energy, yet argmax ties predict a single class and achieve only 12.5% ACC on this balanced test set. A one-hot target requires non-DC graph modes for class discrimination. Lower graph roughness alone therefore cannot define a good classifier; CE, spectrum-resolved errors and held-out decision metrics remain essential.",
    "",
    "## Reproduction",
    "",
    "Run the PowerShell commands in docs/GRAPH_EXPERIMENT.md from the repository root. Exact source shard revision and hashes, dataset CSVs, protocol JSON, per-epoch histories, selected model weights, test probabilities, confusion matrices, environment freeze and test log are preserved locally. Completed runs are skipped by directory; use a fresh output directory or deliberately archive old results before changing the protocol. Mid-run resume is not implemented in the comparison runner; the existing supervised runner retains its checkpoint/resume support.",
    "",
    "## Primary references",
    "",
    "- Original emotion-evaluation paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/f2bb120e9a2cb9c2a50921b7f865c421-Abstract-Conference.html",
    "- Shuman et al., The Emerging Field of Signal Processing on Graphs: https://arxiv.org/abs/1211.0053",
    "- Smola and Kondor, Kernels and Regularization on Graphs: https://people.cs.uchicago.edu/~risi/papers/SmolaKondor.pdf",
    "- Spielman and Srivastava, Graph Sparsification by Effective Resistances: https://arxiv.org/abs/0803.0929",
    "- EmoSet, ICCV 2023 official repository: https://github.com/JingyuanYY/EmoSet",
    "- EmoSet mirror used here: https://huggingface.co/datasets/udeman/EmoSet-118K",
    "",
    "![Graph construction and spectrum](graph_theory.png)",
    "",
    "![Graph heat-filter response](heat_filter.png)",
]
(BASE / "summary.json").write_text(json.dumps(allsummary, indent=2))
(BASE / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
try:
    import markdown

    body = markdown.markdown("\n".join(lines), extensions=["tables", "fenced_code"])
    for figure in BASE.glob("*.png"):
        data = base64.b64encode(figure.read_bytes()).decode("ascii")
        body = body.replace(
            f'src="{figure.name}"', f'src="data:image/png;base64,{data}"'
        )
    style = "body{max-width:1080px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui;color:#17212e}h1,h2{line-height:1.25;color:#123d65}table{border-collapse:collapse;font-size:14px;width:100%;display:block;overflow:auto}th,td{padding:9px;border:1px solid #cad5df;text-align:left}th{background:#eaf2f8}img{max-width:100%;height:auto}code{background:#eef2f5}a{color:#145a9b}"
    (BASE / "REPORT.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Graph signal processing experiment</title><style>'
        + style
        + "</style></head><body>"
        + body
        + "</body></html>",
        encoding="utf8",
    )
except ImportError:
    print("Optional markdown package absent; REPORT.md remains available.")
print(BASE / "REPORT.md")

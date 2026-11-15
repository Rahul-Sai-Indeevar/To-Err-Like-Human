import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("emoset_results")

results = {
    # Supervised results from actual runs
    ("FI", "CE", 0.0):    {"acc": 62.47, "ecc": 73.42, "emc": 48.39, "acc2": 84.70},
    ("FI", "ListMLE", 0.2): {"acc": 62.09, "ecc": 74.39, "emc": 55.92, "acc2": 87.69},
    ("FI", "ListMLE", 1.0): {"acc": 60.05, "ecc": 72.90, "emc": 54.95, "acc2": 87.52},
    ("EmoSet", "CE", 0.0): {"acc": 75.01, "ecc": 83.61, "emc": 59.97, "acc2": 87.69},
    ("EmoSet", "ListMLE", 0.2): {"acc": 75.01, "ecc": 83.61, "emc": 59.97, "acc2": 87.69},
    ("EmoSet", "ListMLE", 1.0): {"acc": 73.87, "ecc": 82.92, "emc": 60.69, "acc2": 93.67},
}

# Paper targets from Table 4
paper_targets = {
    ("FI", "CE", 0.0):    {"acc": 67.3, "ecc": 77.0, "emc": 51.5},
    ("FI", "ListMLE", 0.2): {"acc": 67.3, "ecc": 77.5, "emc": 54.9},
    ("FI", "ListMLE", 1.0): {"acc": 65.7, "ecc": 76.2, "emc": 57.3},
    ("EmoSet", "CE", 0.0): {"acc": 76.3, "ecc": 84.0, "emc": 58.5},
    ("EmoSet", "ListMLE", 0.2): {"acc": 75.5, "ecc": 84.1, "emc": 60.1},
    ("EmoSet", "ListMLE", 1.0): {"acc": 74.5, "ecc": 83.3, "emc": 61.3},
}

ssl_results = [
    {"labels": 800, "rule": "ratio", "acc": 45.77, "ecc": 61.63, "emc": 48.35, "acc2": 77.66},
    {"labels": 4000, "rule": "ratio", "acc": 34.11, "ecc": 53.28, "emc": 48.34, "acc2": 71.01},
    {"labels": 1600, "rule": "ratio", "acc": 32.41, "ecc": 51.30, "emc": 45.38, "acc2": 69.97},
    {"labels": 800, "rule": "fixed", "acc": 27.66, "ecc": 47.99, "emc": 46.09, "acc2": 65.02},
    {"labels": 8000, "rule": "ratio", "acc": 44.87, "ecc": 60.70, "emc": 47.26, "acc2": 75.89},
    {"labels": 8000, "rule": "fixed", "acc": 39.54, "ecc": 56.87, "emc": 46.01, "acc2": 72.68},
]

OUT = Path("docs/plots")
OUT.mkdir(parents=True, exist_ok=True)

# Plot 1: Supervised ACC comparison across datasets and alphas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

datasets = ["FI", "EmoSet"]
alphas = [0.0, 0.2, 1.0]
metrics = ["acc", "ecc", "emc"]
metric_names = ["ACC", "ECC", "EMC"]

for i, metric in enumerate(metrics):
    ax = axes[i]
    x = np.arange(len(alphas))
    width = 0.35

    fi_vals = [results[("FI", "CE", 0.0)][metric] if a == 0.0 else results[("FI", "ListMLE", a)][metric] for a in alphas]
    emoset_vals = [results[("EmoSet", "CE", 0.0)][metric] if a == 0.0 else results[("EmoSet", "ListMLE", a)][metric] for a in alphas]

    ax.bar(x - width/2, fi_vals, width, label="FI", color="#1f77b4", alpha=0.8)
    ax.bar(x + width/2, emoset_vals, width, label="EmoSet", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Alpha")
    ax.set_ylabel(metric_names[i])
    ax.set_title(f"{metric_names[i]} by Alpha")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.0", "0.2", "1.0"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUT / "supervised_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Paper targets vs our results for ACC
fig, ax = plt.subplots(figsize=(10, 6))

datasets = ["FI", "EmoSet"]
alphas = [0.0, 0.2, 1.0]

paper_acc = [paper_targets[("FI", "CE", 0.0)]["acc"], paper_targets[("FI", "ListMLE", 0.2)]["acc"], paper_targets[("FI", "ListMLE", 1.0)]["acc"],
             paper_targets[("EmoSet", "CE", 0.0)]["acc"], paper_targets[("EmoSet", "ListMLE", 0.2)]["acc"], paper_targets[("EmoSet", "ListMLE", 1.0)]["acc"]]
our_acc = [results[("FI", "CE", 0.0)]["acc"], results[("FI", "ListMLE", 0.2)]["acc"], results[("FI", "ListMLE", 1.0)]["acc"],
           results[("EmoSet", "CE", 0.0)]["acc"], results[("EmoSet", "ListMLE", 0.2)]["acc"], results[("EmoSet", "ListMLE", 1.0)]["acc"]]

labels = ["FI\nCE α=0", "FI\nListMLE α=0.2", "FI\nListMLE α=1.0", "EmoSet\nCE α=0", "EmoSet\nListMLE α=0.2", "EmoSet\nListMLE α=1.0"]
x = np.arange(len(labels))
width = 0.35

ax.bar(x - width/2, paper_acc, width, label="Paper Target", color="#2ca02c", alpha=0.8)
ax.bar(x + width/2, our_acc, width, label="Our Results", color="#d62728", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("ACC (%)")
ax.set_title("Paper Targets vs Our Results - ACC")
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(OUT / "paper_vs_ours_acc.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: SSL results comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ssl_metrics = ["acc", "ecc", "emc"]
metric_names_ssl = ["ACC", "ECC", "EMC"]

label_nums = sorted(set(r["labels"] for r in ssl_results))
rules = ["ratio", "fixed"]

for i, metric in enumerate(ssl_metrics):
    ax = axes[i]
    x = np.arange(len(label_nums))
    width = 0.35

    ratio_vals = [next((r[metric] for r in ssl_results if r["labels"] == ln and r["rule"] == "ratio"), 0) for ln in label_nums]
    fixed_vals = [next((r[metric] for r in ssl_results if r["labels"] == ln and r["rule"] == "fixed"), 0) for ln in label_nums]

    ax.bar(x - width/2, ratio_vals, width, label="EMC-adaptive (ratio)", color="#2ca02c", alpha=0.8)
    ax.bar(x + width/2, fixed_vals, width, label="FixMatch (fixed)", color="#d62728", alpha=0.8)

    ax.set_xlabel("Labeled Samples")
    ax.set_ylabel(metric_names_ssl[i])
    ax.set_title(f"{metric_names_ssl[i]} by Label Num & Rule")
    ax.set_xticks(x)
    ax.set_xticklabels([str(ln) for ln in label_nums])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUT / "ssl_results.png", dpi=150, bbox_inches="tight")
plt.close()

print("All plots saved to docs/plots/")
print(f"Files: {[f.name for f in OUT.iterdir()]}")
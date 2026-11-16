import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE_EMOSET = Path("../emoset_results")
BASE_FI = Path("../fi_results")
BASE_FI_SSL = Path("../fi_ssl_results")

results = {}

# Collect results from all directories
all_roots = [BASE_EMOSET, BASE_FI, BASE_FI_SSL]
for base in all_roots:
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        results_file = d / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            test = data.get("test_argmax", {})
            # Normalize key name: remove "_results" suffix if present
            key = d.name.replace("_results", "")
            results[key] = {
                "acc": test.get("acc"),
                "ecc": test.get("ecc"),
                "emc": test.get("emc"),
                "acc2": test.get("acc2"),
                "best_epoch": data.get("best_epoch"),
            }

print("=== Results Summary ===")
print(f"{'Run':<35} {'ACC':>8} {'ECC':>8} {'EMC':>8} {'ACC2':>8}")
for name, r in sorted(results.items()):
    print(f"{name:<35} {r['acc']:.2f}   {r['ecc']:.2f}   {r['emc']:.2f}   {r['acc2']:.2f}")

OUT = Path("docs/plots")
OUT.mkdir(parents=True, exist_ok=True)

# Plot 1: Supervised results - ACC comparison across datasets and alphas
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

alphas = [0.0, 0.2, 1.0]
metrics = ["acc", "ecc", "emc"]
metric_names = ["ACC", "ECC", "EMC"]

for i, metric in enumerate(metrics):
    ax = axes[i]
    x = np.arange(len(alphas))
    width = 0.35

    fi_vals = []
    emoset_vals = []
    for a in alphas:
        # FI runs
        fi_key = f"fi_r50_a{int(a*10):03d}"
        fi_val = results.get(fi_key, {}).get(metric, 0)
        fi_vals.append(fi_val)

        # EmoSet runs - try both naming conventions
        emoset_key = f"emoset_r50_a{int(a*10):03d}"
        emoset_key_alt = f"emoset_r50_a{int(a*10):03d}_results"
        emoset_val = results.get(emoset_key, {}).get(metric, 0)
        if emoset_val == 0:
            emoset_val = results.get(emoset_key_alt, {}).get(metric, 0)
        emoset_vals.append(emoset_val)

    ax.bar(x - width/2, fi_vals, width, label="FI", color="#1f77b4", alpha=0.8)
    ax.bar(x + width/2, emoset_vals, width, label="EmoSet", color="#ff7f0e", alpha=0.8)

    ax.set_xlabel("Alpha")
    ax.set_ylabel(metric_names[i])
    ax.set_title(f"{metric_names[i]} by Alpha (Supervised)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUT / "supervised_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 2: Paper targets vs our results for ACC
fig, ax = plt.subplots(figsize=(10, 6))

paper_acc = [67.3, 67.3, 65.7, 76.3, 75.5, 74.5]  # FI-a0, FI-a0.2, FI-a1.0, EmoSet-a0, EmoSet-a0.2, EmoSet-a1.0
our_acc = [
    results["fi_r50_a0"]["acc"], results["fi_r50_a0.2"]["acc"], results["fi_r50_a1.0"]["acc"],
    results["emoset_r50_a0"]["acc"], results["emoset_r50_a0.2"]["acc"], results["emoset_r50_a1.0"]["acc"],
]

labels = ["FI\nCE α=0", "FI\nListMLE α=0.2", "FI\nListMLE α=1.0", "EmoSet\nCE α=0", "EmoSet\nListMLE α=0.2", "EmoSet\nListMLE α=1.0"]
x = np.arange(len(labels))

ax.bar(x - width/2, paper_acc, width, label="Paper Target", color="#2ca02c", alpha=0.7)
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

label_nums = [800, 8000]
rules = ["ratio", "fixed"]

for i, metric in enumerate(ssl_metrics):
    ax = axes[i]
    x = np.arange(len(label_nums))
    width = 0.35

    ratio_vals = []
    fixed_vals = []
    for ln in label_nums:
        ratio_key = f"emoset_ssl_{ln}_ratio"
        fixed_key = f"emoset_ssl_{ln}_fixed"
        ratio_vals.append(results.get(ratio_key, {}).get(metric, 0))
        fixed_vals.append(results.get(fixed_key, {}).get(metric, 0))

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
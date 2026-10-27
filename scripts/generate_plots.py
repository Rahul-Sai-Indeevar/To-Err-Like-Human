"""Generate loss/metric curves from history.jsonl files for README plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Users\Rahul Sai Indeevar\SP_proj\fi_results")
OUT_DIR = Path(r"C:\Users\Rahul Sai Indeevar\SP_proj\ecc-ver\docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

runs = {
    "alpha=0.0 (CE)": "fi_r50_a0",
    "alpha=0.2 (ListMLE)": "fi_r50_a0.2",
    "alpha=1.0 (ListMLE)": "fi_r50_a1.0",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for label, run_dir in runs.items():
    h = RESULTS_DIR / run_dir / "history.jsonl"
    if not h.exists():
        print(f"Missing: {h}")
        continue
    epochs, loss, acc, ecc, emc = [], [], [], [], []
    with open(h) as f:
        for line in f:
            r = json.loads(line)
            epochs.append(r["epoch"])
            loss.append(r["loss"])
            acc.append(r["val"]["acc"])
            ecc.append(r["val"]["ecc"])
            emc.append(r["val"]["emc"])

    axes[0, 0].plot(epochs, loss, label=label, marker="o", markersize=3)
    axes[0, 1].plot(epochs, acc, label=label, marker="o", markersize=3)
    axes[1, 0].plot(epochs, ecc, label=label, marker="o", markersize=3)
    axes[1, 1].plot(epochs, emc, label=label, marker="o", markersize=3)

axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title("Validation ACC")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("ACC (%)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_title("Validation ECC")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("ECC (%)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_title("Validation EMC")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("EMC (%)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "fi_training_curves.png"
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to {plot_path}")

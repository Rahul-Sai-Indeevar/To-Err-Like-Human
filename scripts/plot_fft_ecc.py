"""Plot clean versus low-pass ECC from archived CSV or a new FFT run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_curve(rows, out):
    clean = next(float(row["ecc_mean"]) for row in rows if row["condition"] == "clean")
    filtered = sorted((float(row["condition"].split("_")[1]), float(row["ecc_mean"]))
                      for row in rows if row["condition"].startswith("lowpass_"))
    cutoff, ecc = zip(*filtered)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(cutoff, ecc, "o-", color="#096D71", label="Low-pass images")
    ax.axhline(clean, color="#A34D30", linestyle="--", label="Clean images")
    ax.set(xlabel="Low-pass cutoff (cycles/pixel)", ylabel="Mean test ECC (%)",
           title="Spatial low-pass filtering on frozen EmoSet models")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/fft/summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("runs/fft_lowpass/lowpass_ecc.png"))
    args = parser.parse_args()
    if args.input.suffix == ".csv":
        with args.input.open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = json.loads(args.input.read_text(encoding="utf8"))["results"]
    plot_curve(rows, args.out)


if __name__ == "__main__":
    main()

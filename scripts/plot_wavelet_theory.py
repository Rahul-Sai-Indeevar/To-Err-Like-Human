"""Reproducible synthetic illustrations; no training images or GPU required."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_figures(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:160, :160]
    image = (.35 + .22 * xx / 159 + .15 * np.sin(2 * np.pi * yy / 90)
             + .12 * np.sin(2 * np.pi * xx / 7) * np.cos(2 * np.pi * yy / 9)
             + .08 * (((xx // 2 + yy // 2) % 2) * 2 - 1))

    def projection(x):
        return x.reshape(40, 4, 40, 4).mean((1, 3)).repeat(4, 0).repeat(4, 1)

    low = projection(image)
    detail = image - low
    commutator = projection(np.roll(image, 1, axis=1)) - np.roll(low, 1, axis=1)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for axis, data, title, signed in zip(axes, (image, low, detail, commutator),
        ("Synthetic input x", "LL2 reconstruction Px", "Signed details (I−P)x", "P(T₁x) − T₁(Px)"),
        (False, False, True, True)):
        limit = .4 if signed else 1
        view = axis.imshow(data, cmap="RdBu_r" if signed else "gray", vmin=-limit if signed else 0, vmax=limit)
        axis.set_title(title, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(view, ax=axis, fraction=.046, pad=.03)
    fig.suptitle("Exact coarse/detail partition; one-pixel grid sensitivity", fontsize=14)
    fig.text(.5, .015, "Fixed display ranges. Signed details are never clipped or rescaled before model input. Synthetic signal only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(out / "wavelet_partition.png", dpi=180)
    plt.close(fig)

    f = np.linspace(0, .5, 1001)
    omega = 2 * np.pi * f
    h = (1 + np.exp(-1j * omega)) / np.sqrt(2)
    g = (1 - np.exp(-1j * omega)) / np.sqrt(2)
    h_coarse = h * (1 + np.exp(-2j * omega)) / np.sqrt(2)
    g_coarse = h * (1 - np.exp(-2j * omega)) / np.sqrt(2)
    powers = (abs(h_coarse)**2 / 4, abs(g_coarse)**2 / 4, abs(g)**2 / 2)
    if not np.allclose(sum(powers), 1, atol=1e-14):
        raise ValueError("Weighted analysis-band power identity failed")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for power, name, color in zip(powers, ("Coarse LL2 (1-D)", "Scale-2 detail (1-D)", "Scale-1 detail (1-D)"), ("#176b87", "#d68a25", "#9b4971")):
        axes[0].plot(f, power, label=name, color=color, linewidth=2)
    axes[0].plot(f, sum(powers), color="black", linestyle="--", label="Sum = 1", linewidth=1)
    axes[0].set(xlabel="Spatial frequency (cycles/pixel)", ylabel="Fraction of complex-sinusoid energy", ylim=(-.03, 1.06), title="Two-level 1-D Haar analysis bank")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].bar(("Low projection", "All details"), (1 / 16, 15 / 16), color=("#176b87", "#9b4971"))
    axes[1].set(ylim=(0, 1), ylabel="Expected fraction of total noise energy", title="J=2 two-dimensional iid white noise")
    for i, value in enumerate((1 / 16, 15 / 16)):
        axes[1].text(i, value + .02, f"{value:.4f}", ha="center")
    fig.text(.5, .012, "Left: analysis-filter energy including coefficient sampling rates, not an LTI transfer function for the shift-variant projection.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out / "wavelet_filter_theory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/wavelet_study"))
    make_figures(parser.parse_args().out)

"""CPU diagnostics for the repository Haar filter bank.

Run via the repository's Python environment:
  python scripts/run_wavelet_sp_checks.py --output runs/haar_checks

The image diagnostic uses existing prepared test manifests and local images.
Only aggregate measurements are released. No classifiers are fitted/evaluated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
import yaml

from ecc.experiment_data import atomic_json
from torchvision.transforms import Resize

from ecc.wavelets import HaarBands, haar_analysis2d, haar_synthesis2d, wavelet_components


WIDTH = 64
BLOCK = 4
SIGMA = 0.03
DATASET_LABELS = {"fi": "FI", "emoset": "EmoSet subset", "artphoto": "ArtPhoto"}
BANDS = ("ll", "lh", "hl", "hh")
OFFSETS = tuple(range(1 - BLOCK, BLOCK))
TAPS = tuple((BLOCK - abs(k)) / BLOCK**2 for k in OFFSETS)


def energy(x):
    return x.square().sum(dim=(-3, -2, -1))


def max_abs(x):
    return x.abs().max().item()


def complex_tone(fx, fy=0.0):
    axis = torch.arange(WIDTH, dtype=torch.float64)
    phase = 2 * torch.pi * (fy * axis[:, None] + fx * axis[None, :])
    return torch.stack((torch.cos(phase), torch.sin(phase)))[None, ...]


def fft_coefficient(x, column_bin):
    z = torch.complex(x[0, 0, 0, :], x[0, 1, 0, :])
    return (torch.fft.fft(z) / z.numel())[column_bin].item()


def project(x):
    return wavelet_components(x, levels=2)[0]


def phase_average(x):
    """Average aligned reconstructions over all 16 phase origins."""
    output = torch.zeros_like(x)
    for row in range(BLOCK):
        for column in range(BLOCK):
            shifted = x.roll((row, column), dims=(-2, -1))
            output += project(shifted).roll((-row, -column), dims=(-2, -1))
    return output / BLOCK**2


def triangular_filter(x):
    """Equivalent separable, circular convolution for all phase origins."""
    row_filtered = sum(tap * x.roll(k, dims=-2) for k, tap in zip(OFFSETS, TAPS))
    return sum(tap * row_filtered.roll(k, dims=-1) for k, tap in zip(OFFSETS, TAPS))


def triangle_response(frequency):
    omega = 2 * np.pi * np.asarray(frequency)
    return sum(tap * np.cos(k * omega) for k, tap in zip(OFFSETS, TAPS))


def run_tone_sweep():
    sweep = {}
    max_errors = defaultdict(float)
    for name, factor in (("columns_only", 0), ("diagonal", 1), ("rows_only", -1)):
        rows = []
        for k in range(WIDTH // 2 + 1):
            f = k / WIDTH
            fx, fy = (f, factor * f) if factor >= 0 else (0.0, f)
            tone = complex_tone(fx, fy)
            coeffs = haar_analysis2d(tone)
            input_energy = energy(tone).item()
            low_x, low_y = np.cos(np.pi * fx)**2, np.cos(np.pi * fy)**2
            high_x, high_y = np.sin(np.pi * fx)**2, np.sin(np.pi * fy)**2
            pred = dict(zip(BANDS, (
                low_y * low_x, low_y * high_x, high_y * low_x, high_y * high_x,
            )))
            measured = {b: energy(getattr(coeffs, b)).item() / input_energy for b in BANDS}
            expected_ll2 = float(triangle_response(fx) * triangle_response(fy))
            measured_ll2 = energy(project(tone)).item() / input_energy
            triangle = triangular_filter(tone)
            max_errors["band_energy_fraction"] = max(
                max_errors["band_energy_fraction"], *(abs(pred[b] - measured[b]) for b in BANDS)
            )
            max_errors["two_level_coarse_energy_fraction"] = max(
                max_errors["two_level_coarse_energy_fraction"], abs(expected_ll2 - measured_ll2)
            )
            max_errors["phase_avg_gain_max_pixel_error"] = max(
                max_errors["phase_avg_gain_max_pixel_error"], max_abs(triangle - expected_ll2 * tone)
            )
            max_errors["perfect_reconstruction_max_pixel_error"] = max(
                max_errors["perfect_reconstruction_max_pixel_error"],
                max_abs(haar_synthesis2d(coeffs) - tone),
            )
            rows.append({
                "frequency": f, "fx": fx, "fy": fy,
                "predicted": pred, "measured": measured,
                "predicted_ll2_energy_fraction": expected_ll2,
                "measured_ll2_energy_fraction": measured_ll2,
                "measured_phase_avg_amplitude_gain": (triangle * tone).sum().item() / input_energy,
            })
        sweep[name] = rows
    return {"grid_size": WIDTH, "rows_per_sweep": len(rows), "sweeps": sweep, "max_errors": max_errors}


def run_alias_check():
    f = 1 / 8
    tone, folded = complex_tone(f), complex_tone(f + 0.5)
    coeffs = haar_analysis2d(tone)
    zero = torch.zeros_like(coeffs.ll)
    low = haar_synthesis2d(HaarBands(coeffs.ll, zero, zero, zero))
    high = haar_synthesis2d(HaarBands(zero, coeffs.lh, zero, zero))
    full = haar_synthesis2d(coeffs)
    base_index = round(f * WIDTH)
    image_index = base_index + WIDTH // 2
    coefficients = {}
    for name, signal in (("low_only", low), ("high_only", high), ("full", full)):
        coefficients[name] = {}
        for label, k in (("base", base_index), ("folded", image_index)):
            value = fft_coefficient(signal, k)
            coefficients[name][label] = [value.real, value.imag]
    return {
        "input_frequency": f,
        "folded_frequency": f + 0.5,
        "folded_frequency_signed": f - 0.5,
        "downsampled_frequency_cycles_per_output_sample": 2 * f,
        "max_even_sample_difference_between_input_and_folded_tones": max_abs(
            tone[..., ::2] - folded[..., ::2]
        ),
        "max_reconstruction_error": max_abs(full - tone),
        "coefficients": coefficients,
        "predicted_base_low": float(np.cos(np.pi * f)**2),
        "predicted_base_high": float(np.sin(np.pi * f)**2),
        "predicted_folded_low_imag": float(np.sin(2 * np.pi * f) / 2),
        "predicted_folded_high_imag": float(-np.sin(2 * np.pi * f) / 2),
    }


def run_cycle_checks(seed):
    impulse = torch.zeros((1, 1, WIDTH, WIDTH), dtype=torch.float64)
    impulse[..., 0, 0] = 1
    noise = torch.randn((24, 1, 128, 128), dtype=torch.float64,
                        generator=torch.Generator().manual_seed(seed))
    direct = phase_average(noise[:1])
    convolved = triangular_filter(noise[:1])
    # A bin at each offset from the origin wraps on the circular grid.
    rolled_indices = torch.as_tensor([k % WIDTH for k in OFFSETS])
    expected_kernel = torch.outer(torch.tensor(TAPS, dtype=torch.float64),
                                  torch.tensor(TAPS, dtype=torch.float64))
    expected_impulse = torch.zeros_like(impulse)
    expected_impulse[..., rolled_indices[:, None], rolled_indices[None, :]] = expected_kernel
    phase_impulse = phase_average(impulse)
    response = torch.fft.fft2(phase_impulse)[0, 0]
    frequency = np.arange(WIDTH) / WIDTH
    expected_response = torch.as_tensor(np.outer(triangle_response(frequency),
                                                triangle_response(frequency)))
    # Steps at two off-grid locations make the averaging-grid effect visible.
    step = torch.zeros_like(impulse)
    step[..., 21:45] = 1
    fixed, invariant = project(step), triangular_filter(step)
    return {
        "phase_count": BLOCK**2,
        "offsets": OFFSETS,
        "kernel_1d": TAPS,
        "impulse_error": max_abs(phase_impulse - expected_impulse),
        "fft_response_error": max_abs(response - expected_response),
        "explicit_phase_average_vs_convolution_max_error": max_abs(direct - convolved),
        "white_noise_power_fraction": {
            "fixed_predicted": 1 / BLOCK**2,
            "phase_avg_predicted": sum(tap**2 for tap in TAPS)**2,
            "fixed_measured": (energy(project(noise)).sum() / energy(noise).sum()).item(),
            "phase_avg_measured": (energy(triangular_filter(noise)).sum() / energy(noise).sum()).item(),
        },
        "step": {
            "columns": list(range(14, 33)),
            "input": step[0, 0, 0, 14:33].tolist(),
            "fixed": fixed[0, 0, 0, 14:33].tolist(),
            "phase_avg": invariant[0, 0, 0, 14:33].tolist(),
        },
    }


def choose_test_rows(root, generator, per_class):
    manifest = root / "test.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty prepared test manifest: {manifest}")
    groups = defaultdict(list)
    for row_id, row in enumerate(rows):
        groups[row["label"]].append(row_id)
    picked = []
    counts = {}
    for label, ids in sorted(groups.items()):
        count = min(len(ids), per_class)
        counts[label] = count
        picked.extend(generator.choice(ids, count, replace=False).tolist())
    selected = sorted(picked)
    return rows, selected, counts, hashlib.sha256(manifest.read_bytes()).hexdigest()


def edge_energy(x):
    # The wrap seam is excluded from the image-gradient energy measurement.
    return energy(x.diff(dim=-1)) + energy(x.diff(dim=-2))


def run_real_images(datasets, seed, per_class):
    summary = {}
    selection_rng = np.random.default_rng(seed)
    noise_rng = torch.Generator().manual_seed(seed + 1)
    resize = Resize((192, 192))
    for key, name in DATASET_LABELS.items():
        root = Path(datasets[key]).resolve()
        rows, selected, counts, csv_digest = choose_test_rows(root, selection_rng, per_class)
        measurements = {method: defaultdict(list) for method in ("fixed", "phase_avg")}
        for first in range(0, len(selected), 12):
            decoded = []
            for row_id in selected[first:first + 12]:
                image = (root / rows[row_id]["path"]).resolve()
                if not image.is_relative_to(root):
                    raise ValueError(f"Path escapes prepared dataset: {image}")
                with Image.open(image) as source:
                    rgb = resize(ImageOps.exif_transpose(source).convert("RGB"))
                    decoded.append(np.asarray(rgb)[16:176, 16:176].transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(decoded)).to(torch.float64) / 255
            shift = x.roll(1, dims=-1)
            centered = energy(x - x.mean(dim=(-1, -2), keepdim=True))
            edges = edge_energy(x)
            white = SIGMA * torch.randn(x.shape, dtype=x.dtype, generator=noise_rng)
            noise_power = energy(white)
            for method, transform in (("fixed", project), ("phase_avg", triangular_filter)):
                y = transform(x)
                commutator = transform(shift) - y.roll(1, dims=-1)
                measures = {
                    "shift_commutator_rms_rel_centered": (energy(commutator) / centered.clamp(min=1e-30)).sqrt(),
                    "clean_distortion_rms_rel_centered": (energy(y - x) / centered.clamp(min=1e-30)).sqrt(),
                    "gradient_energy_fraction": edge_energy(y) / edges.clamp(min=1e-30),
                    "noise_energy_fraction": energy(transform(white)) / noise_power,
                }
                for key, value in measures.items():
                    measurements[method][key].extend(value.tolist())
        summary[name] = {
            "split": f"{key}/test.csv",
            "manifest_sha256": csv_digest,
            "manifest_size": len(rows),
            "images": len(selected),
            "images_per_class": counts,
            "selected_zero_based_rows": selected,
            "methods": {
                method: {
                    key: {"mean": float(np.mean(values)), "median": float(np.median(values))}
                    for key, values in values_by_key.items()
                } for method, values_by_key in measurements.items()
            },
        }
        print(f"{name}: measured {len(selected)} local test images", flush=True)
    return summary


def check_identities(tone, alias, cycle):
    # Errors are computed in float64 and catch a changed sign, axis, or scale.
    errors = list(tone["max_errors"].values()) + [
        alias["max_even_sample_difference_between_input_and_folded_tones"],
        alias["max_reconstruction_error"],
        abs(alias["coefficients"]["low_only"]["folded"][1] - alias["predicted_folded_low_imag"]),
        abs(alias["coefficients"]["high_only"]["folded"][1] - alias["predicted_folded_high_imag"]),
        abs(complex(*alias["coefficients"]["full"]["folded"])),
        cycle["impulse_error"], cycle["fft_response_error"],
        cycle["explicit_phase_average_vs_convolution_max_error"],
    ]
    if max(errors) > 1e-11:
        raise RuntimeError(f"Haar model and measurement disagree: max error {max(errors):.6g}")


def make_figure(result, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ink = "#142C38"
    teal, orange = "#096D71", "#A34D30"
    sweeps = result["tone_sweep"]["sweeps"]
    f = [r["frequency"] for r in sweeps["columns_only"]]
    for band, color in (("ll", teal), ("lh", orange)):
        d = sweeps["columns_only"]
        axes[0, 0].plot(f, [r["predicted"][band] for r in d], c=color, label=f"{band.upper()} theory")
        axes[0, 0].scatter(f[::3], [r["measured"][band] for r in d[::3]],
                           s=23, facecolors="white", edgecolors=color, label=f"{band.upper()} measured")
    axes[0, 0].set(title="Horizontal tone: predicted and measured band energy",
                  xlabel="Cycles per input pixel", ylabel="Fraction of input energy")
    axes[0, 0].legend(ncols=2)
    coeff = result["alias"]["coefficients"]
    names = ["low_only", "high_only", "full"]
    axes[0, 1].bar(["Low branch", "High branch", "Recombined"],
                   [coeff[name]["folded"][1] for name in names], color=[teal, orange, ink])
    axes[0, 1].set(title="The two alias components cancel",
                  ylabel="Imaginary FFT coefficient at folded bin")
    axes[0, 1].axhline(0, linewidth=0.75, c=ink)
    step = result["cycle_spinning"]["step"]
    for label, key, color in (("Input", "input", ink), ("Fixed Haar", "fixed", orange),
                              ("16-phase average", "phase_avg", teal)):
        axes[1, 0].step(step["columns"], step[key], label=label, c=color, where="mid")
    axes[1, 0].set(title="The shift-invariant average blurs a synthetic edge",
                   xlabel="Pixel column", ylabel="Signal amplitude")
    axes[1, 0].legend()
    real = result["datasets"]
    if not real:
        fig.delaxes(axes[1, 1])
        fig.savefig(output / "wavelet_sp_checks.png", dpi=180)
        plt.close(fig)
        return
    categories = list(real)
    i = np.arange(len(categories))
    for method, shift, color, label in (
        ("fixed", -.18, orange, "Fixed Haar"), ("phase_avg", .18, teal, "16-phase average"),
    ):
        means = [real[key]["methods"][method]["noise_energy_fraction"]["mean"] for key in categories]
        axes[1, 1].bar(i + shift, np.array(means) * 100, width=.36, color=color, label=label)
    axes[1, 1].set(title="Controlled Gaussian noise on local image crops",
                   ylabel="Noise energy retained (%)", xticks=i, xticklabels=categories)
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.savefig(output / "wavelet_sp_checks.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wavelet_study.yaml"))
    parser.add_argument("--synthetic-only", action="store_true", help="Run identities without images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--per-class", type=int, default=24)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    torch.set_num_threads(2)
    with torch.inference_mode():
        tone = run_tone_sweep()
        alias = run_alias_check()
        cycle = run_cycle_checks(args.seed)
        check_identities(tone, alias, cycle)
        data = {}
        if not args.synthetic_only:
            datasets = yaml.safe_load(args.config.read_text(encoding="utf8"))["datasets"]
            data = run_real_images(datasets, args.seed, args.per_class)
    result = {
        "protocol": {
            "seed": args.seed, "per_class_cap": args.per_class,
            "sampling_rule": "Test manifest, uniformly without replacement within each class; all if fewer",
            "image": "EXIF-corrected RGB, bilinear resize 192×192, center crop 160×160, float64 in [0,1]",
            "boundary": "Circular for explicit shifts and phase average; no wrap in real image gradients",
            "noise": f"Zero-mean Gaussian white noise, sigma {SIGMA}, matched across transforms, no clipping",
            "shift_columns": 1,
            "source": "ecc.wavelets (haar_analysis2d, haar_synthesis2d, wavelet_components)",
            "python_version": platform.python_version(), "torch_version": torch.__version__,
            "private_source_note": "Prepared research test images stay local; no photographs are redistributed.",
        },
        "tone_sweep": tone, "alias": alias, "cycle_spinning": cycle, "datasets": data,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "results.json", result)
    make_figure(result, args.output)
    print(f"Saved theory and measurement evidence to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

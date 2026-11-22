"""Inference across paired training seeds, with optional crossed image bootstrap.

Training seeds are the experimental units for seed uncertainty. Per-image
predictions are never counted as independent replications of model training.
All intervals are descriptive when a protocol was chosen after seeing results.
"""
from __future__ import annotations

import numpy as np


def _finite_vector(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return x


def exact_paired_signflip(differences):
    """Two-sided exact sign-flip p-value for the absolute mean difference.

This tests exchangeability of treatment labels within seed pairs (equivalently
sign symmetry under the null). It is not assumption-free for arbitrary seed
distributions. All 2**n assignments, including the observed one, are counted;
no Monte Carlo correction is appropriate for an exhaustive enumeration.
"""
    d = _finite_vector(differences, "differences")
    n = len(d)
    if n > 20:
        raise ValueError("Exact sign-flip enumeration is limited to 20 seed pairs")
    observed = abs(float(d.sum()))
    # Sum rather than mean avoids unnecessary rounding; a relative tolerance
    # handles assignments tied with the observed statistic.
    tolerance = 32 * np.finfo(float).eps * max(float(np.abs(d).sum()), 1e-300)
    extreme = 0
    total = 1 << n
    bits = np.arange(n, dtype=np.uint64)
    for start in range(0, total, 65536):
        assignments = np.arange(start, min(start + 65536, total), dtype=np.uint64)
        signs = 2.0 * ((assignments[:, None] >> bits) & 1).astype(float) - 1.0
        stats = np.abs(signs @ d)
        extreme += int(np.count_nonzero(stats >= observed - tolerance))
    return {"p_two_sided": float(extreme / total), "assignments": total,
            "assumption": "Treatment-label exchangeability within seed pairs / sign symmetry under the null"}


def paired_seed_summary(baseline, candidate):
    """Candidate-minus-baseline mean, sample SD, t interval, and exact test.

The Student interval assumes independent seed pairs and an approximately
normal distribution of their differences. With few pairs this is fragile.
SciPy supplies the Student quantile; when unavailable the interval is omitted.
"""
    a = _finite_vector(baseline, "baseline")
    b = _finite_vector(candidate, "candidate")
    if a.shape != b.shape:
        raise ValueError("baseline and candidate must contain the same paired seeds")
    d = b - a
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else None
    interval = None
    interval_note = "Undefined with one seed pair"
    if n > 1:
        try:
            from scipy.stats import t
        except ImportError:
            interval_note = "SciPy unavailable: Student interval omitted"
        else:
            half = float(t.ppf(0.975, df=n - 1)) * sd / np.sqrt(n)
            interval = [float(d.mean() - half), float(d.mean() + half)]
            interval_note = "Student t95 over independent seed-pair differences; normality assumption, fragile for small n"
    leave_one_out = (d.sum() - d) / (n - 1) if n > 1 else None
    return {
        "n_seed_pairs": n,
        "differences": d.tolist(),
        "mean_difference": float(d.mean()),
        "sample_sd_difference": sd,
        "standard_error_difference": sd / np.sqrt(n) if n > 1 else None,
        "t95": interval,
        "t95_note": interval_note,
        "exact_signflip": exact_paired_signflip(d) if n <= 20 else None,
        "positive_pairs": int((d > 0).sum()),
        "negative_pairs": int((d < 0).sum()),
        "zero_pairs": int((d == 0).sum()),
        "leave_one_out_mean_range": [float(leave_one_out.min()), float(leave_one_out.max())]
        if leave_one_out is not None else None,
        "unit": "paired independently trained seed, not test image",
    }


def crossed_seed_image_bootstrap(paired_image_differences, labels, repeats=2000, seed=20261117):
    """Bootstrap an additive metric over both seeds and stratified test images.

Input is candidate-minus-baseline per-image metric contribution, shape
(seed_pairs, images). It supports mean accuracy or ECC contributions in the
caller's chosen units, not nonlinear metrics such as EMC or macro F1. Sample
seed pairs with replacement, and independently sample images with replacement
within each class. The SAME sampled image indices apply to every sampled seed
and method; this is a crossed bootstrap, not independent within-seed images.

This does not create additional training seeds, account for alternative train
datasets, or support a population-wide claim from a small fixed dataset.
"""
    d = np.asarray(paired_image_differences, dtype=np.float64)
    y = np.asarray(labels)
    if d.ndim != 2 or min(d.shape) < 1 or not np.isfinite(d).all():
        raise ValueError("paired_image_differences must be finite with shape (seed_pairs, images)")
    if y.ndim != 1 or len(y) != d.shape[1]:
        raise ValueError("labels must match the image axis")
    if y.dtype.kind not in "iu" or np.any(y < 0):
        raise ValueError("labels must be nonnegative integers")
    if not isinstance(repeats, (int, np.integer)) or repeats < 100:
        raise ValueError("repeats must be an integer of at least 100")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    draws = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        selected_seeds = rng.integers(0, d.shape[0], size=d.shape[0])
        averaged_seeds = d[selected_seeds].mean(axis=0)
        total = 0.0
        for indices in groups:
            chosen = rng.choice(indices, size=len(indices), replace=True)
            total += float(averaged_seeds[chosen].sum())
        draws[i] = total / d.shape[1]
    return {
        "mean_difference": float(d.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "n_seed_pairs": d.shape[0], "n_images": d.shape[1],
        "repeats": int(repeats), "rng_seed": int(seed),
        "definition": "Crossed paired-seed and shared class-stratified image bootstrap for an additive mean metric",
        "limitation": "Exploratory percentile interval; few seeds give unstable uncertainty; no new training-dataset variation",
    }

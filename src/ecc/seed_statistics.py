"""Inference across paired training seeds.

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

import numpy as np
import pytest

from ecc.seed_statistics import exact_paired_signflip, paired_seed_summary, crossed_seed_image_bootstrap


def test_exact_signflip_exhaustive_known_values_and_scale_invariance():
    assert exact_paired_signflip([1, 2, 3])["p_two_sided"] == 0.25
    assert exact_paired_signflip([1e-15, 2e-15, 3e-15])["p_two_sided"] == 0.25
    assert exact_paired_signflip([0, 0, 0])["p_two_sided"] == 1
    assert exact_paired_signflip([1, -1])["p_two_sided"] == 1
    assert exact_paired_signflip([1])["p_two_sided"] == 1
    with pytest.raises(ValueError):
        exact_paired_signflip(np.ones(21))


def test_seed_summary_t_interval_and_leave_one_out():
    result = paired_seed_summary([4, 10, 20], [5, 12, 23])
    assert result["differences"] == [1, 2, 3]
    assert result["mean_difference"] == 2
    assert result["sample_sd_difference"] == 1
    assert result["leave_one_out_mean_range"] == [1.5, 2.5]
    if result["t95"] is not None:
        assert result["t95"] == pytest.approx([-0.4841377117, 4.4841377117])
    single = paired_seed_summary([5], [6])
    assert single["t95"] is None
    assert single["sample_sd_difference"] is None


def test_crossed_bootstrap_keeps_seed_uncertainty_when_images_repeated():
    # A million duplicate images must not become a million training replications.
    d = np.array([[-1] * 40, [1] * 40], dtype=float)
    result = crossed_seed_image_bootstrap(d, np.repeat([0, 1], 20), repeats=1000)
    assert result["mean_difference"] == 0
    assert result["ci95"] == [-1, 1]
    assert result["bootstrap_sd"] > 0.5


def test_crossed_bootstrap_preserves_class_counts_and_is_deterministic():
    # Within each class every contribution is identical; stratification fixes
    # the mean exactly, regardless of selected image identities.
    d = np.tile([0, 0, 10], (3, 1))
    y = np.array([0, 0, 1])
    first = crossed_seed_image_bootstrap(d, y, repeats=100)
    second = crossed_seed_image_bootstrap(d, y, repeats=100)
    assert first == second
    assert first["ci95"] == pytest.approx([10 / 3, 10 / 3])
    assert first["bootstrap_sd"] < 1e-12


def test_invalid_inputs():
    for a, b in [([], []), ([1], [1, 2]), ([np.nan], [0]), ([[1]], [0])]:
        with pytest.raises(ValueError):
            paired_seed_summary(a, b)
    with pytest.raises(ValueError):
        crossed_seed_image_bootstrap([[1, 2]], [0])
    with pytest.raises(ValueError):
        crossed_seed_image_bootstrap([[1, 2]], [0, 1], repeats=10)
    with pytest.raises(ValueError):
        crossed_seed_image_bootstrap([[1, 2]], [0., 1.])


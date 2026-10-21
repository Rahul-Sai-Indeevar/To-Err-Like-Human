from ecc.ssl import adaptive_threshold


def test_threshold_decreases_with_emc_and_is_clipped():
    # Eq. 5: tau' = tau * eps / emc. eps in (0,1), so higher emc -> lower tau.
    # emc == eps leaves tau unchanged.
    t = [adaptive_threshold(0.95, 0.5, e) for e in (0.25, 0.5, 0.9)]
    assert t == sorted(t, reverse=True)  # non-increasing as EMC grows
    assert max(t) <= 0.98 and min(t) >= 0.7
    # EMC = eps -> tau' = tau (no change, no clipping).
    assert adaptive_threshold(0.95, 0.5, 0.5) == 0.95
    # EMC -> 0 -> tau' -> +inf -> clipped to tau_max.
    assert adaptive_threshold(0.95, 0.5, 0.0001) == 0.98
    # EMC = 1 -> 0.95 * 0.5 / 1 = 0.475 -> clipped to tau_min.
    assert adaptive_threshold(0.95, 0.5, 1.0) == 0.7


def test_fixed_rule_is_fixmatch():
    assert adaptive_threshold(0.95, 0.5, 0.2, rule="fixed") == 0.95

from ecc.ssl import adaptive_threshold


def test_threshold_decreases_with_emc_and_is_clipped():
    t = [adaptive_threshold(0.95, 0.5, e) for e in (0.3, 0.5, 0.7, 0.9)]
    assert t == sorted(t, reverse=True)
    assert max(t) <= 0.98 and min(t) >= 0.7
    assert adaptive_threshold(0.95, 0.5, 1.0) == 0.7
    assert adaptive_threshold(0.95, 0.5, 0.0001) == 0.98


def test_fixed_rule_is_fixmatch():
    assert adaptive_threshold(0.95, 0.5, 0.2, rule="fixed") == 0.95

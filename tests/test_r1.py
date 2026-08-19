from neuroscan_ood.experiments.r1 import summarise


def test_summarise_computes_gap_ci_and_test():
    r = {"random": [0.99, 0.98, 0.97], "grouped": [0.90, 0.89, 0.91]}
    s = summarise(r)
    assert abs(s["random_mean"] - 0.98) < 1e-9
    assert abs(s["grouped_mean"] - 0.90) < 1e-9
    assert abs(s["gap"] - 0.08) < 1e-9
    assert s["n"] == 3
    assert s["all_positive"] is True
    # keys the report and CSV depend on
    for k in ("random_ci", "grouped_ci", "gap_ci", "t", "p"):
        assert k in s
    lo, hi = s["gap_ci"]
    assert lo <= s["gap"] <= hi
    assert 0.0 <= s["p"] <= 1.0

from neuroscan_ood.experiments.r1 import summarise


def test_summarise_computes_gap():
    r = {"random": [0.99, 0.98, 0.97], "grouped": [0.90, 0.89, 0.91]}
    s = summarise(r)
    assert abs(s["random_mean"] - 0.98) < 1e-9
    assert abs(s["grouped_mean"] - 0.90) < 1e-9
    assert abs(s["gap"] - 0.08) < 1e-9

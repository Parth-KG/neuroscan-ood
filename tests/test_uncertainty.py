import torch

from neuroscan_ood.experiments.uncertainty import acc_at_coverage, fit_temperature, referral_curve


def test_referral_prefers_confident_and_matches_known_case():
    # confidences 0.9,0.8 correct; 0.7,0.6 wrong -> keeping top half is perfect, full set is 50%
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    labels = torch.tensor(
        [0, 0, 0, 0]
    )  # top-2 predict class 0 (right); bottom-2 predict class 1 (wrong)
    cov, acc = referral_curve(probs, labels)
    assert abs(acc_at_coverage(cov, acc, 0.5) - 1.0) < 1e-9
    assert abs(acc_at_coverage(cov, acc, 1.0) - 0.5) < 1e-9


def test_fit_temperature_returns_positive_and_softens_overconfident():
    # very sharp (over-confident) logits, half of them wrong -> best T should be > 1
    logits = torch.tensor([[6.0, 0.0], [6.0, 0.0], [0.0, 6.0], [0.0, 6.0]])
    labels = torch.tensor([0, 1, 0, 1])  # half wrong
    T = fit_temperature(logits, labels)
    assert T > 1.0

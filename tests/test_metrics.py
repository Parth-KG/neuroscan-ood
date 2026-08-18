import torch

from neuroscan_ood.eval.metrics import compute_metrics


def test_perfect_predictions():
    logits = torch.tensor([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
    m = compute_metrics(logits, torch.tensor([0, 1, 2]), num_classes=3)
    assert abs(m["accuracy"] - 1.0) < 1e-6
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_all_wrong_predictions():
    logits = torch.tensor([[0.0, 10, 0], [0, 0, 10.0], [10.0, 0, 0]])
    m = compute_metrics(logits, torch.tensor([0, 1, 2]), num_classes=3)
    assert m["accuracy"] == 0.0

import numpy as np
import torch
import torch.nn as nn

from neuroscan_ood.experiments.mitigate import adapt_bn, normalize_intensity


def test_normalize_intensity_is_deterministic_and_changes_brightness():
    a = (np.random.RandomState(0).rand(32, 32) * 120 + 60).astype("uint8")  # mid-band only
    out1 = normalize_intensity(a)
    out2 = normalize_intensity(a)
    assert np.array_equal(out1, out2)  # deterministic
    assert out1.dtype == np.uint8 and out1.shape == a.shape
    assert out1.max() - out1.min() > int(a.max()) - int(a.min())  # equalisation widens the range


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.fc(x.mean(dim=(2, 3)))


def test_adapt_bn_changes_bn_stats_only():
    torch.manual_seed(0)
    net = _Net().eval()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    # data with a clearly non-unit distribution so BN stats must move
    xs = [(torch.randn(8, 3, 8, 8) * 5 + 3, torch.zeros(8, dtype=torch.long)) for _ in range(3)]
    adapt_bn(net, xs, torch.device("cpu"))
    after = net.state_dict()
    # BN running stats changed
    assert not torch.allclose(before["bn.running_mean"], after["bn.running_mean"])
    assert not torch.allclose(before["bn.running_var"], after["bn.running_var"])
    # every non-BN-stat parameter is untouched
    for k in ["conv.weight", "conv.bias", "bn.weight", "bn.bias", "fc.weight", "fc.bias"]:
        assert torch.allclose(before[k], after[k]), k


def test_matched_normalizer_is_full_range_and_deterministic():
    import numpy as np

    from neuroscan_ood.experiments.mitigate import normalize_intensity_matched

    a = (np.random.RandomState(1).rand(16, 16) * 100 + 40).astype("uint8")  # compressed band
    o1 = normalize_intensity_matched(a)
    o2 = normalize_intensity_matched(a)
    assert np.array_equal(o1, o2)
    assert o1.min() == 0 and o1.max() == 255  # re-stretched to the training convention

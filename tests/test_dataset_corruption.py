import numpy as np
import pandas as pd
import torch
from PIL import Image

from neuroscan_ood.data.dataset import MriDataset


def test_corruption_none_matches_identity(tmp_path):
    img = (np.random.RandomState(0).rand(40, 40) * 255).astype("uint8")
    Image.fromarray(img).save(tmp_path / "x.png")
    df = pd.DataFrame([{"filename": "x.png", "label": 0}])
    ds_none = MriDataset(df, tmp_path, 32, train=False)  # RGB path
    ds_id = MriDataset(df, tmp_path, 32, train=False, corruption=lambda a: a)  # L->identity->RGB
    assert torch.allclose(ds_none[0][0], ds_id[0][0], atol=1e-6)

"""torch Dataset over a manifest split, plus the transform builder.

An optional `corruption` callable (uint8 HxW array -> uint8 HxW array) supports the controlled
acquisition-shift study (R2). When it is None the image is loaded exactly as before, so Phase 1
and R1 results are unaffected.
"""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool):
    # All images are resized here, so native resolution (512 vs 256) needs no special handling.
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class MriDataset(Dataset):
    def __init__(self, df, images_root, image_size: int, train: bool, corruption=None):
        self.df = df.reset_index(drop=True)
        self.images_root = str(images_root)
        self.tf = build_transforms(image_size, train)
        self.corruption = corruption

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = os.path.join(self.images_root, r["filename"])
        if self.corruption is None:
            img = Image.open(path).convert("RGB")  # grayscale MRI -> 3 channels
        else:
            arr = np.array(Image.open(path).convert("L"))
            arr = self.corruption(arr)
            img = Image.fromarray(arr).convert("RGB")
        return self.tf(img), int(r["label"])

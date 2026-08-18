"""Deterministic seeding across python/numpy/torch."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Without these, cuDNN chooses nondeterministic kernels and re-runs diverge.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    # Each DataLoader worker must be reseeded or shuffle order varies across runs.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

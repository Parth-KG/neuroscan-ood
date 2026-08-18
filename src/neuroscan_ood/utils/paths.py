"""Path resolution from a single root. No logic beyond building paths."""

import os
from pathlib import Path


def get_root() -> Path:
    root = os.environ.get("NEUROSCAN_ROOT")
    if not root:
        raise RuntimeError("NEUROSCAN_ROOT is not set; point it at the durable project folder.")
    return Path(root)


def prepared_root() -> Path:
    return get_root() / "data" / "prepared"


def images_root() -> Path:
    return prepared_root() / "images"


def runs_root() -> Path:
    return get_root() / "runs"


def run_dir(run_id: str) -> Path:
    return runs_root() / run_id

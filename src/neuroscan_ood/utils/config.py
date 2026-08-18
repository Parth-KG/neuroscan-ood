"""Load and lightly validate a YAML run config."""

from pathlib import Path

import yaml

REQUIRED_KEYS = ["seed", "source", "split", "classes", "model", "train", "run_id"]


def load_config(path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    return cfg


def save_config(cfg: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(public, f, sort_keys=False)

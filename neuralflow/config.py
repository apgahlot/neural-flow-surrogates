"""Configuration loading and lightweight attribute access."""
from __future__ import annotations
import os
import yaml


class Config(dict):
    """A dict that also supports attribute access and nested dotted lookup."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def get_path(self, dotted, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str) -> Config:
    """Load a YAML config file into a Config object."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw)


def ensure_dirs(cfg: Config) -> None:
    """Create the project storage directories if they do not exist."""
    for key in ("data_dir", "ckpt_dir", "results_dir"):
        os.makedirs(cfg.paths[key], exist_ok=True)

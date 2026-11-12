"""Configuration loading. Paths in config.yaml are resolved against the repo root."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH_KEYS = ("db_path", "raw_dir", "output_dir", "log_dir")


def _resolve(path_str: str) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def load_config(path: str | Path | None = None, mode: str = "live") -> dict:
    """Load config.yaml. ``mode='demo'`` swaps in the synthetic database and output folder."""
    if mode not in {"live", "demo"}:
        raise ValueError(f"mode must be 'live' or 'demo', got {mode!r}")
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = copy.deepcopy(yaml.safe_load(fh))

    if mode == "demo":
        demo = cfg.get("demo", {})
        cfg["project"]["db_path"] = demo.get("db_path", "data/demo_synthetic.db")
        cfg["project"]["output_dir"] = demo.get("output_dir", "outputs/demo_synthetic")
        cfg["fundamentals"]["source"] = "synthetic"

    for key in PATH_KEYS:
        cfg["project"][key] = _resolve(cfg["project"][key])
    cfg["universe"]["file"] = _resolve(cfg["universe"]["file"])
    cfg["mode"] = mode
    return cfg


def ensure_dirs(cfg: dict) -> None:
    p = cfg["project"]
    Path(p["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    for key in ("raw_dir", "output_dir", "log_dir"):
        Path(p[key]).mkdir(parents=True, exist_ok=True)
    (Path(p["output_dir"]) / "charts").mkdir(parents=True, exist_ok=True)

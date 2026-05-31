"""Centralized path resolution for data, outputs, and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def get_data_dir() -> Path:
    """
    Dataset root containing per-task folders (pdbbind_v2016, zinc, ...).

    Search order:
    1. MULTIGEODTA_DATA_DIR
    2. ./data
    3. ./create_dataset (legacy layout from original repo / HF download)
    """
    env = os.environ.get("MULTIGEODTA_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    root = get_project_root()
    for candidate in (root / "data", root / "create_dataset"):
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return root / "data"


def get_output_dir(subpath: str | None = None) -> Path:
    """Checkpoint and log root. Default: ./outputs (legacy: ./MultiGeoDTA/output)."""
    env = os.environ.get("MULTIGEODTA_OUTPUT_DIR")
    if env:
        base = Path(env).expanduser().resolve()
    else:
        root = get_project_root()
        legacy = root / "MultiGeoDTA" / "output"
        base = legacy if legacy.exists() else root / "outputs"
    if subpath:
        return base / subpath
    return base


def resolve_checkpoint_dir(
    output_dir: str | Path | None = None,
    model_file: str | None = None,
) -> Path:
    """
    Resolve directory containing checkpoint_*.pt files.

    Matches original layout: ``{MULTIGEODTA_OUTPUT_DIR}/{model_file}/checkpoint_N.pt``
    When ``output_dir`` already contains checkpoints, use it directly.
    """
    base = get_output_dir()
    if model_file:
        candidate = Path(model_file)
        if candidate.is_absolute():
            return candidate if candidate.is_dir() else candidate.parent
        # Nested path e.g. pdbbind_v2021_similarity/new_new/0.5
        ckpt_root = base / model_file
        if (ckpt_root / "checkpoint_1.pt").exists() or ckpt_root.is_dir():
            return ckpt_root
        if output_dir:
            od = Path(output_dir)
            if (od / "checkpoint_1.pt").exists():
                return od
        return ckpt_root
    return Path(output_dir) if output_dir else base


def task_data_dir(task_name: str) -> Path:
    """Per-task data folder under data root."""
    return get_data_dir() / task_name

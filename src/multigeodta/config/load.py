"""YAML config loading and CLI merge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from multigeodta.utils.paths import get_project_root


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: Optional[str] = None,
    defaults_path: Optional[str] = None,
) -> Dict[str, Any]:
    root = get_project_root()
    defaults = load_yaml(defaults_path or root / "configs" / "base.yaml")
    if config_path:
        overrides = load_yaml(Path(config_path))
        defaults.update(overrides)
    return defaults


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, help="YAML config (merged over defaults)")
    parser.add_argument("--task", help="Task name")
    parser.add_argument(
        "--split_method",
        default=None,
        choices=["new_compound", "new_protein", "new_new"],
    )
    parser.add_argument("--thre", default=None, choices=["0.3", "0.4", "0.5", "0.6"])
    parser.add_argument(
        "--variant",
        default=None,
        help="Robustness variant for pdbbind_v2016_robustness (e.g. noised_scale_0.2)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--contact_cutoff", type=float, default=None)
    parser.add_argument("--num_rbf", type=int, default=None)
    parser.add_argument("--mlp_dims", type=int, nargs="+", default=None)
    parser.add_argument("--mlp_dropout", type=float, default=None)
    parser.add_argument("--n_ensembles", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--monitor_metric", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_file", default=None, help="Checkpoint subdir for evaluate/screen")
    parser.add_argument("--data_dir", default=None, help="Override MULTIGEODTA_DATA_DIR")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny data subset for quick sanity check",
    )
    parser.add_argument("--save_log", action="store_true", default=None)
    parser.add_argument("--no_save_log", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true", default=None)
    parser.add_argument("--no_save_checkpoint", action="store_true")
    parser.add_argument("--save_prediction", action="store_true", default=None)
    parser.add_argument("--no_save_prediction", action="store_true")


def merge_cli_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Overlay non-None CLI values onto config."""
    mapping = {
        "task": "task",
        "split_method": "split_method",
        "thre": "thre",
        "variant": "variant",
        "seed": "seed",
        "contact_cutoff": "contact_cutoff",
        "num_rbf": "num_rbf",
        "mlp_dims": "mlp_dims",
        "mlp_dropout": "mlp_dropout",
        "n_ensembles": "n_ensembles",
        "batch_size": "batch_size",
        "n_epochs": "n_epochs",
        "patience": "patience",
        "eval_freq": "eval_freq",
        "lr": "lr",
        "monitor_metric": "monitor_metric",
        "output_dir": "output_dir",
        "model_file": "model_file",
        "data_dir": "data_dir",
        "device": "device",
        "smoke": "smoke",
    }
    for arg_name, key in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            config[key] = val
    if getattr(args, "parallel", False):
        config["parallel"] = True
    if getattr(args, "smoke", False):
        config["smoke"] = True
    if getattr(args, "no_save_log", False):
        config["save_log"] = False
    elif getattr(args, "save_log", None):
        config["save_log"] = True
    if getattr(args, "no_save_checkpoint", False):
        config["save_checkpoint"] = False
    elif getattr(args, "save_checkpoint", None):
        config["save_checkpoint"] = True
    if getattr(args, "no_save_prediction", False):
        config["save_prediction"] = False
    elif getattr(args, "save_prediction", None):
        config["save_prediction"] = True
    if args.config and not config.get("task"):
        task_cfg = load_yaml(Path(args.config))
        config.update(task_cfg)
    return config

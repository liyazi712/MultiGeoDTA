"""Unified CLI: multigeodta train | evaluate | screen"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from multigeodta.config.load import add_train_args, load_config, merge_cli_args
from multigeodta.inference.virtual_screen import VirtualScreenRunner
from multigeodta.training.experiment import TrainingExperiment
from multigeodta.utils.paths import get_output_dir


def run_train(config: dict) -> None:
    exp = TrainingExperiment(config)
    if config.get("save_log") or config.get("save_checkpoint") or config.get("save_prediction"):
        exp.saver.save_config(config, "args.yaml")
    exp.train(
        n_epochs=config.get("n_epochs"),
        patience=config.get("patience"),
        eval_freq=config.get("eval_freq", 1),
        monitoring_score=config.get("monitor_metric", "mse"),
    )
    exp.test(
        save_prediction=config.get("save_prediction", True),
        test_tag="Ensemble model",
        print_log=True,
    )


def run_evaluate(config: dict) -> None:
    if not config.get("model_file"):
        raise ValueError("--model_file is required for evaluate")
    exp = TrainingExperiment(config)
    if config.get("save_log") or config.get("save_checkpoint") or config.get("save_prediction"):
        exp.saver.save_config(config, "args.yaml")
    exp.test_saved(
        model_file=config["model_file"],
        save_prediction=config.get("save_prediction", True),
    )


def run_screen(config: dict) -> None:
    if not config.get("model_file"):
        raise ValueError("--model_file is required for virtual screening")
    config.setdefault("checkpoint_root", str(get_output_dir()))
    runner = VirtualScreenRunner(config)
    runner.run(model_file=config["model_file"])


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="multigeodta",
        description="MultiGeo-DTA: multimodal drug-target binding affinity",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train ensemble models")
    add_train_args(train_p)
    train_p.set_defaults(_run="train")

    eval_p = sub.add_parser("evaluate", help="Evaluate saved checkpoints")
    add_train_args(eval_p)
    eval_p.set_defaults(_run="evaluate")

    screen_p = sub.add_parser("screen", help="Virtual screening (ZINC)")
    add_train_args(screen_p)
    screen_p.set_defaults(_run="screen")

    args = parser.parse_args(argv)
    config = load_config(getattr(args, "config", None))
    if args.config:
        with open(args.config) as f:
            config.update(yaml.safe_load(f) or {})
    config = merge_cli_args(config, args)

    if not config.get("task") and args._run != "screen":
        parser.error("--task is required (or set in --config)")

    if args._run == "train":
        run_train(config)
    elif args._run == "evaluate":
        run_evaluate(config)
    elif args._run == "screen":
        config.setdefault("task", "zinc")
        run_screen(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())

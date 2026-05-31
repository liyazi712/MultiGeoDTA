"""Logging, checkpoint I/O, and early stopping."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


class Logger:
    def __init__(self, logfile: Path | None = None, level: int = logging.INFO):
        self.logger = logging.getLogger("multigeodta")
        self.logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s\t%(message)s", "%Y-%m-%d %H:%M:%S")
        for hd in self.logger.handlers[:]:
            self.logger.removeHandler(hd)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        if logfile is not None:
            logfile = Path(logfile)
            logfile.parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(logfile, "w")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


class Saver:
    def __init__(self, output_dir: str | Path):
        self.save_dir = Path(output_dir)

    def mkdir(self) -> None:
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def save_ckp(self, state_dict, filename: str = "checkpoint.pt") -> None:
        self.mkdir()
        torch.save(state_dict, str(self.save_dir / filename))

    def save_df(self, df, filename: str, index: bool = False, float_format: str = "%.6f") -> None:
        self.mkdir()
        df.to_csv(self.save_dir / filename, float_format=float_format, index=index, sep="\t")

    def save_config(self, config: dict, filename: str) -> None:
        self.mkdir()
        with open(self.save_dir / filename, "w") as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 100,
        eval_freq: int = 1,
        best_score: float | None = None,
        delta: float = 1e-9,
        higher_better: bool = True,
    ):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False

    def not_improved(self, val_score: float) -> bool:
        if np.isnan(val_score):
            return True
        if self.higher_better:
            return val_score < self.best_score + self.delta
        return val_score > self.best_score - self.delta

    def update(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return True
        if self.not_improved(val_score):
            self.counter += self.eval_freq
            if self.patience is not None and self.counter > self.patience:
                self.early_stop = True
            return False
        self.best_score = val_score
        self.counter = 0
        return True

"""Virtual screening on unlabeled compound libraries (e.g. ZINC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from multigeodta.data.tasks.registry import build_task
from multigeodta.models.dta_model import DTAModel
from multigeodta.utils.logging import Logger, Saver
from multigeodta.utils.paths import resolve_checkpoint_dir


class VirtualScreenRunner:
    """Load trained checkpoints and score unlabeled data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs/zinc"))
        self.n_ensembles = config.get("n_ensembles", 5)
        self.batch_size = config.get("batch_size", 128)
        self.device_id = config.get("device", 0)
        self.saver = Saver(self.output_dir)
        self.logger = Logger(
            logfile=self.saver.save_dir / "exp.log" if config.get("save_log", True) else None
        )
        self.dataset = build_task("zinc", config)
        self.devices = [
            torch.device(f"cuda:{self.device_id}") for _ in range(self.n_ensembles)
        ]
        self.model_config = {
            "mlp_dims": config.get("mlp_dims", [1024, 512]),
            "mlp_dropout": config.get("mlp_dropout", 0.25),
        }
        self._loaders = None
        self._split_df = None

    def _get_loader(self) -> DataLoader:
        if self._loaders is None:
            data, split_df = self.dataset.get_split()
            self._split_df = split_df
            self._loaders = DataLoader(
                dataset=data["test"],
                batch_size=self.batch_size,
                collate_fn=data["test"].collate,
                shuffle=False,
                drop_last=False,
                num_workers=int(self.config.get("num_workers", 8)),
            )
        return self._loaders

    def run(
        self,
        model_file: str,
        save_df_name: str = "prediction.tsv",
    ) -> pd.DataFrame:
        ckpt_dir = resolve_checkpoint_dir(
            self.config.get("checkpoint_root", self.output_dir),
            model_file,
        )
        loader = self._get_loader()
        esb_yp = None

        for midx in range(self.n_ensembles):
            model = DTAModel(**self.model_config).to(self.devices[midx])
            path = ckpt_dir / f"checkpoint_{midx + 1}.pt"
            model.load_state_dict(torch.load(path, map_location=self.devices[midx]))
            yp = self._predict(model, loader, self.devices[midx], midx + 1)
            esb_yp = yp.reshape(1, -1) if esb_yp is None else np.vstack((esb_yp, yp.reshape(1, -1)))

        y_pred_avg = np.mean(esb_yp, axis=0)
        df = self._split_df["test"].copy()
        df["y_pred_avg"] = y_pred_avg
        for i in range(self.n_ensembles):
            df[f"y_pred_{i + 1}"] = esb_yp[i]
        self.saver.save_df(df, save_df_name, float_format="%g")
        self.logger.info(f"Saved predictions to {self.saver.save_dir / save_df_name}")
        return df

    @staticmethod
    def _predict(model, loader, device, midx: int) -> np.ndarray:
        model.eval()
        yp = torch.Tensor()
        with torch.cuda.device(device):
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Screening model {midx}"):
                    xd = batch["drug"].to(device)
                    xp = batch["protein"].to(device)
                    protein_seq = batch["full_seq"].to(device)
                    pocket_seq = batch["poc_seq"].to(device)
                    smile_seq = batch["smile_seq"].to(device)
                    yh, _, _, _, _ = model(xd, xp, protein_seq, pocket_seq, smile_seq)
                    yp = torch.cat([yp, yh.detach().cpu()], dim=0)
        return yp.view(-1).numpy()


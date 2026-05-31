"""High-level training and evaluation experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from multigeodta.data.tasks.registry import build_task
from multigeodta.metrics.regression import evaluation_metrics
from multigeodta.models.dta_model import DTAModel
from multigeodta.training.trainer import evaluate_loader, train_one_ensemble
from multigeodta.utils.logging import EarlyStopping, Logger, Saver
from multigeodta.utils.paths import resolve_checkpoint_dir


class TrainingExperiment:
    """Train and evaluate MultiGeo-DTA on labeled benchmarks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.task = config["task"]
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.save_checkpoint = config.get("save_checkpoint", True)
        self.parallel = config.get("parallel", False)
        self.n_ensembles = config.get("n_ensembles", 5)
        self.n_epochs = config.get("n_epochs", 100)
        self.batch_size = config.get("batch_size", 128)
        self.lr = config.get("lr", 1e-4)
        self.device_id = config.get("device", 0)

        self.saver = Saver(self.output_dir)
        self.logger = Logger(
            logfile=self.saver.save_dir / "exp.log" if config.get("save_log", True) else None
        )

        if self.task == "pdbbind_v2021_similarity":
            self.logger.info(
                f"split_method={config.get('split_method')} thre={config.get('thre')}"
            )

        self.dataset = build_task(self.task, config)
        self._split = None
        self._loaders = None
        self.devices = [
            torch.device(f"cuda:{self.device_id}") for _ in range(self.n_ensembles)
        ]
        self.model_config = {
            "mlp_dims": config.get("mlp_dims", [1024, 512]),
            "mlp_dropout": config.get("mlp_dropout", 0.25),
        }
        self.criterion = F.mse_loss
        self.build_models()
        self.logger.info(str(self.models[0]))

    def build_models(self) -> None:
        self.models = [
            DTAModel(**self.model_config).to(self.devices[i])
            for i in range(self.n_ensembles)
        ]
        self.optimizers = [
            optim.Adam(m.parameters(), lr=self.lr) for m in self.models
        ]

    def _loader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            shuffle=shuffle,
            drop_last=False,
        )

    @property
    def task_loaders(self) -> Dict[str, DataLoader]:
        if self._loaders is None:
            data, _ = self.dataset.get_split()
            self._loaders = {
                s: self._loader(data[s], shuffle=(s == "train")) for s in data
            }
        return self._loaders

    @property
    def task_dfs(self):
        if self._split is None:
            _, self._split = self.dataset.get_split()
        return self._split

    def train(
        self,
        n_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        eval_freq: int = 1,
        monitoring_score: str = "mse",
        test_after_train: bool = True,
    ) -> None:
        n_epochs = n_epochs or self.n_epochs
        loaders = self.task_loaders
        rets = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(train_one_ensemble)(
                midx=i + 1,
                model=self.models[i],
                optimizer=self.optimizers[i],
                train_loader=loaders["train"],
                valid_loader=loaders["valid"],
                test_loader=loaders["test"],
                device=self.devices[i],
                n_epochs=n_epochs,
                eval_freq=eval_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion,
                logger=self.logger,
                patience=patience or self.config.get("patience", 20),
                save_checkpoint=self.save_checkpoint,
                output_dir=self.output_dir,
                test_after_train=test_after_train,
            )
            for i in range(self.n_ensembles)
        )
        for r in rets:
            self.models[r["midx"] - 1] = r["model"]

    def _format_predict_df(self, results, esb_yp=None, test_df=None):
        df = self.task_dfs["test"].copy() if test_df is None else test_df.copy()
        df["y_pred_avg"] = results.get("y_pred_avg", results.get("y_pred"))
        if esb_yp is not None:
            for i in range(self.n_ensembles):
                df[f"y_pred_{i + 1}"] = esb_yp[i]
        return df

    def test(
        self,
        save_prediction: bool = False,
        save_df_name: str = "prediction.tsv",
        test_tag: str = "Ensemble model",
        print_log: bool = True,
    ) -> None:
        loader = self.task_loaders["test"]
        rets_list = []
        for i, model in enumerate(self.models):
            rets_list.append(
                evaluate_loader(
                    model=model,
                    loader=loader,
                    device=self.devices[i],
                    loss_fn=self.criterion,
                    midx=i + 1,
                    test_tag=f"Model {i+1}",
                    print_log=True,
                    logger=self.logger,
                )
            )
        esb_yp = np.vstack([r["y_pred"].reshape(1, -1) for r in rets_list])
        y_true = rets_list[-1]["y_true"]
        y_pred = np.mean(esb_yp, axis=0)
        metrics = evaluation_metrics(
            y_true, y_pred, eval_metrics=["rmse", "mae", "spearman", "pearson"]
        )
        if print_log:
            self.logger.info(
                f"{test_tag} | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )
        results = {"y_true": y_true, "y_pred": y_pred, "y_pred_avg": y_pred}
        df = self._format_predict_df(results, esb_yp=esb_yp)
        if save_prediction:
            self.saver.save_df(df, save_df_name, float_format="%g")

    def test_saved(
        self,
        model_file: Optional[str] = None,
        save_prediction: bool = False,
        save_df_name: str = "prediction.tsv",
    ) -> None:
        ckpt_dir = resolve_checkpoint_dir(self.output_dir, model_file)
        metric_names = ["rmse", "mae", "pearson", "spearman", "ci", "rm2"]
        loader = self.task_loaders["test"]
        rets_list = []
        model_metrics = []

        for midx in range(self.n_ensembles):
            model = DTAModel(**self.model_config).to(self.devices[midx])
            path = ckpt_dir / f"checkpoint_{midx + 1}.pt"
            model.load_state_dict(torch.load(path, map_location=self.devices[midx]))
            rets = evaluate_loader(
                model=model,
                loader=loader,
                device=self.devices[midx],
                loss_fn=self.criterion,
                midx=midx + 1,
                test_tag=f"Model {midx + 1}",
                print_log=True,
                logger=self.logger,
            )
            rets_list.append(rets)
            model_metrics.append(
                pd.Series(
                    evaluation_metrics(
                        rets["y_true"], rets["y_pred"], eval_metrics=metric_names
                    ),
                    name=f"Model {midx + 1}",
                )
            )

        esb_yp = np.vstack([r["y_pred"].reshape(1, -1) for r in rets_list])
        y_pred_avg = np.mean(esb_yp, axis=0)
        y_true = rets_list[-1]["y_true"]
        final_metrics = evaluation_metrics(y_true, y_pred_avg, eval_metrics=metric_names)

        all_df = pd.concat(model_metrics, axis=1)
        all_df["Mean"] = all_df.mean(axis=1)
        all_df["Std"] = all_df.std(axis=1)
        all_df["Ensemble"] = pd.Series(final_metrics)
        print(all_df)
        self.saver.save_df(all_df, "all_model_metrics.tsv", float_format="%g", index=metric_names)

        results = {"y_pred_avg": y_pred_avg, "y_pred": y_pred_avg}
        df = self._format_predict_df(results, esb_yp=esb_yp)
        if save_prediction:
            self.saver.save_df(df, save_df_name, float_format="%g")

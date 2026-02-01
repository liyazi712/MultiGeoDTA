"""
Training Module for MultiGeoDTA

Provides a unified interface for training and evaluating DTA models.
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple, Any

from multigeodta.models import DTAModel
from multigeodta.data import DTADataset
from multigeodta.utils import Logger, Saver, EarlyStopping, evaluation_metrics


def init_weights(m: nn.Module):
    """Initialize model weights using Kaiming initialization."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DTATrainer:
    """
    Trainer class for Drug-Target Affinity prediction models.

    Handles the complete training workflow including:
    - Model initialization and ensemble management
    - Training loop with validation
    - Early stopping
    - Model checkpointing
    - Evaluation and prediction

    Args:
        model_config: Configuration for DTAModel
        n_ensembles: Number of ensemble models
        batch_size: Training batch size
        lr: Learning rate
        device: CUDA device ID or 'cpu'
        output_dir: Directory for saving outputs
        save_log: Whether to save training logs
        save_checkpoint: Whether to save model checkpoints

    Example:
        >>> trainer = DTATrainer(n_ensembles=5, device=0)
        >>> trainer.setup_data(train_data, valid_data, test_data)
        >>> trainer.train(n_epochs=100)
        >>> results = trainer.evaluate()
    """

    def __init__(self,
                 model_config: Optional[Dict] = None,
                 n_ensembles: int = 5,
                 batch_size: int = 128,
                 lr: float = 0.0001,
                 device: int = 0,
                 output_dir: str = './output',
                 save_log: bool = True,
                 save_checkpoint: bool = True,
                 parallel: bool = False):

        self.n_ensembles = n_ensembles
        self.batch_size = batch_size
        self.lr = lr
        self.parallel = parallel
        self.output_dir = output_dir
        self.save_checkpoint = save_checkpoint

        # Setup devices
        if isinstance(device, int):
            self.devices = [torch.device(f'cuda:{device}') for _ in range(n_ensembles)]
        else:
            self.devices = [torch.device(device) for _ in range(n_ensembles)]

        # Model configuration
        self.model_config = model_config or {
            'mlp_dims': [1024, 512],
            'mlp_dropout': 0.25
        }

        # Initialize saver and logger
        self.saver = Saver(output_dir)
        self.logger = Logger(
            logfile=self.saver.save_dir / 'exp.log' if save_log else None
        )

        # Initialize models
        self.models = None
        self.optimizers = None
        self.criterion = F.mse_loss

        # Data loaders
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self._build_models()

    def _build_models(self):
        """Initialize ensemble models and optimizers."""
        self.models = [
            DTAModel(**self.model_config).to(self.devices[i])
            for i in range(self.n_ensembles)
        ]
        self.optimizers = [
            optim.Adam(model.parameters(), lr=self.lr)
            for model in self.models
        ]
        self.logger.info(f"Built {self.n_ensembles} ensemble models")
        self.logger.info(f"Model parameters: {self.models[0].get_num_parameters():,}")

    def setup_data(self, train_dataset: DTADataset,
                   valid_dataset: DTADataset,
                   test_dataset: DTADataset):
        """
        Setup data loaders from datasets.

        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            test_dataset: Test dataset
        """
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            collate_fn=train_dataset.collate, shuffle=True, drop_last=False
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=self.batch_size,
            collate_fn=valid_dataset.collate, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size,
            collate_fn=test_dataset.collate, shuffle=False
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Valid samples: {len(valid_dataset)}")
        self.logger.info(f"Test samples: {len(test_dataset)}")

    def train(self, n_epochs: int = 100, patience: int = 20,
              eval_freq: int = 1, monitoring_score: str = 'mse',
              test_after_train: bool = True) -> List[Dict]:
        """
        Train ensemble models.

        Args:
            n_epochs: Maximum number of training epochs
            patience: Patience for early stopping
            eval_freq: Epochs between evaluations
            monitoring_score: Metric to monitor for early stopping
            test_after_train: Whether to evaluate on test set after training

        Returns:
            List of training results for each model
        """
        if self.train_loader is None:
            raise RuntimeError("Data not setup. Call setup_data() first.")

        results_list = []

        for i in range(self.n_ensembles):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training Model {i+1}/{self.n_ensembles}")
            self.logger.info(f"{'='*50}")

            result = self._train_single_model(
                model_idx=i,
                n_epochs=n_epochs,
                patience=patience,
                eval_freq=eval_freq,
                monitoring_score=monitoring_score,
                test_after_train=test_after_train
            )
            results_list.append(result)

        return results_list

    def _train_single_model(self, model_idx: int, n_epochs: int,
                           patience: int, eval_freq: int,
                           monitoring_score: str,
                           test_after_train: bool) -> Dict:
        """Train a single model from the ensemble."""
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]
        device = self.devices[model_idx]

        model.to(device)
        model.train()
        model.apply(init_weights)

        stopper = EarlyStopping(patience=patience, eval_freq=eval_freq, higher_better=False)
        best_model_state = None

        for epoch in range(1, n_epochs + 1):
            # Training
            total_loss = 0
            total_loss_pred = 0
            total_loss_ps = 0
            total_loss_cs = 0

            for batch in self.train_loader:
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                protein_seq = batch['full_seq'].to(device)
                pocket_seq = batch['poc_seq'].to(device)
                smile_seq = batch['smile_seq'].to(device)
                y = batch['y'].to(device)

                optimizer.zero_grad()

                yh, protein_feats, seq_feats, compound_feats, smile_feats = model(
                    xd, xp, protein_seq, pocket_seq, smile_seq
                )

                # Multi-task loss
                loss_pred = self.criterion(yh, y.view(-1, 1))
                loss_ps = self.criterion(protein_feats, seq_feats)
                loss_cs = self.criterion(compound_feats, smile_feats)
                loss = loss_pred + 10 * loss_ps + 10 * loss_cs

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_pred += loss_pred.item()
                total_loss_ps += loss_ps.item()
                total_loss_cs += loss_cs.item()

            train_loss = total_loss / len(self.train_loader)

            # Validation
            if epoch % eval_freq == 0:
                val_results = self._evaluate_single_model(model, device, self.valid_loader)
                is_best = stopper.update(val_results['metrics'][monitoring_score])

                if is_best:
                    best_model_state = copy.deepcopy(model.state_dict())

                self.logger.info(
                    f"M-{model_idx+1} E-{epoch} | "
                    f"Train Loss: {train_loss:.3f} | "
                    f"Valid Loss: {val_results['loss']:.3f} | " +
                    ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()]) +
                    f" | Best {monitoring_score}: {stopper.best_score:.4f}"
                )

            if stopper.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Test evaluation
        if test_after_train:
            test_results = self._evaluate_single_model(model, device, self.test_loader)
            self.logger.info(
                f"Model {model_idx+1} Test | Loss: {test_results['loss']:.4f} | " +
                ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
            )

        # Save checkpoint
        if self.save_checkpoint:
            self.saver.save_checkpoint(
                best_model_state or model.state_dict(),
                f'checkpoint_{model_idx+1}.pt'
            )

        return {'model_idx': model_idx, 'model': model}

    def _evaluate_single_model(self, model: nn.Module, device: torch.device,
                               data_loader: DataLoader) -> Dict:
        """Evaluate a single model on a dataset."""
        model.eval()
        yt, yp = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                protein_seq = batch['full_seq'].to(device)
                pocket_seq = batch['poc_seq'].to(device)
                smile_seq = batch['smile_seq'].to(device)
                y = batch['y'].to(device)

                yh, protein_feats, seq_feats, compound_feats, smile_feats = model(
                    xd, xp, protein_seq, pocket_seq, smile_seq
                )

                loss_pred = self.criterion(yh, y.view(-1, 1))
                loss_ps = self.criterion(protein_feats, seq_feats)
                loss_cs = self.criterion(compound_feats, smile_feats)
                loss = loss_pred + 10 * loss_ps + 10 * loss_cs
                total_loss += loss.item()

                yp.extend(yh.cpu().numpy().flatten())
                yt.extend(y.cpu().numpy().flatten())

        model.train()

        yt = np.array(yt)
        yp = np.array(yp)

        metrics_result = evaluation_metrics(
            yt, yp, eval_metrics=['mse', 'spearman', 'pearson']
        )

        return {
            'y_true': yt,
            'y_pred': yp,
            'loss': total_loss / len(data_loader),
            'metrics': metrics_result
        }

    def evaluate(self, save_prediction: bool = True,
                 save_name: str = 'prediction.tsv') -> Dict:
        """
        Evaluate ensemble models on test set.

        Args:
            save_prediction: Whether to save predictions
            save_name: Filename for predictions

        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_losses = []

        for i, model in enumerate(self.models):
            results = self._evaluate_single_model(
                model, self.devices[i], self.test_loader
            )
            all_predictions.append(results['y_pred'])
            all_losses.append(results['loss'])

            self.logger.info(
                f"Model {i+1} Test | " +
                ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()])
            )

        # Ensemble prediction (average)
        y_true = results['y_true']
        y_pred_ensemble = np.mean(all_predictions, axis=0)
        ensemble_loss = np.mean(all_losses)

        # Ensemble metrics
        ensemble_metrics = evaluation_metrics(
            y_true, y_pred_ensemble,
            eval_metrics=['rmse', 'mae', 'pearson', 'spearman', 'ci', 'rm2']
        )

        self.logger.info(
            f"\nEnsemble Test | Loss: {ensemble_loss:.4f} | " +
            ' | '.join([f'{k}: {v:.4f}' for k, v in ensemble_metrics.items()])
        )

        results = {
            'y_true': y_true,
            'y_pred': y_pred_ensemble,
            'loss': ensemble_loss,
            'metrics': ensemble_metrics,
            'individual_predictions': all_predictions
        }

        if save_prediction:
            df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred_ensemble,
            })
            for i, pred in enumerate(all_predictions):
                df[f'y_pred_{i+1}'] = pred
            self.saver.save_df(df, save_name)

        return results

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            data_loader: DataLoader for prediction data

        Returns:
            Ensemble predictions
        """
        all_predictions = []

        for i, model in enumerate(self.models):
            model.eval()
            predictions = []

            with torch.no_grad():
                for batch in data_loader:
                    xd = batch['drug'].to(self.devices[i])
                    xp = batch['protein'].to(self.devices[i])
                    protein_seq = batch['full_seq'].to(self.devices[i])
                    pocket_seq = batch['poc_seq'].to(self.devices[i])
                    smile_seq = batch['smile_seq'].to(self.devices[i])

                    yh, _, _, _, _ = model(xd, xp, protein_seq, pocket_seq, smile_seq)
                    predictions.extend(yh.cpu().numpy().flatten())

            all_predictions.append(predictions)

        return np.mean(all_predictions, axis=0)

    def load_checkpoints(self, checkpoint_dir: str):
        """
        Load model checkpoints.

        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        for i in range(self.n_ensembles):
            checkpoint_path = checkpoint_dir / f'checkpoint_{i+1}.pt'
            state_dict = torch.load(checkpoint_path)
            self.models[i].load_state_dict(state_dict)
            self.models[i].to(self.devices[i])
            self.logger.info(f"Loaded checkpoint for model {i+1}")

    def save_config(self, config: Dict, filename: str = 'config.yaml'):
        """Save training configuration."""
        self.saver.save_config(config, filename)

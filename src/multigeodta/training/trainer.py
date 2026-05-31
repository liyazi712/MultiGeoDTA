"""Low-level train/eval loops for a single ensemble member."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from multigeodta.metrics.regression import evaluation_metrics
from multigeodta.utils.logging import EarlyStopping, Logger


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def unpack_model_outputs(outputs):
    """Consistent unpack: affinity, protein, seq, compound, smile features."""
    return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]


def compute_loss(model, batch, device, loss_fn):
    xd = batch["drug"].to(device)
    xp = batch["protein"].to(device)
    protein_seq = batch["full_seq"].to(device)
    pocket_seq = batch["poc_seq"].to(device)
    smile_seq = batch["smile_seq"].to(device)
    y = batch["y"].to(device).float().view(-1, 1)

    yh, protein_feats, seq_feats, compound_feats, smile_feats = unpack_model_outputs(
        model(xd, xp, protein_seq, pocket_seq, smile_seq)
    )
    loss_pred = loss_fn(yh, y)
    loss_ps = loss_fn(protein_feats, seq_feats)
    loss_cs = loss_fn(compound_feats, smile_feats)
    loss = loss_pred + 10 * loss_ps + 10 * loss_cs
    return loss, loss_pred, loss_ps, loss_cs, yh, y


def train_one_ensemble(
    *,
    midx: int,
    model: nn.Module,
    optimizer,
    train_loader,
    valid_loader,
    test_loader,
    device: torch.device,
    n_epochs: int,
    eval_freq: int,
    monitoring_score: str,
    loss_fn,
    logger: Logger,
    patience: Optional[int] = 20,
    save_checkpoint: bool = False,
    output_dir: Optional[Path] = None,
    test_after_train: bool = True,
) -> Dict[str, Any]:
    stopper = EarlyStopping(patience=patience, eval_freq=eval_freq, higher_better=False)
    best_state = None

    model = model.to(device)
    model.train()
    model.apply(init_weights)

    with torch.cuda.device(device):
        for epoch in range(1, n_epochs + 1):
            total_loss = total_pred = total_ps = total_cs = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                loss, lp, lps, lcs, _, _ = compute_loss(model, batch, device, loss_fn)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_pred += lp.item()
                total_ps += lps.item()
                total_cs += lcs.item()

            n_batch = len(train_loader)
            train_loss = total_loss / n_batch
            if epoch % eval_freq == 0:
                val_results = evaluate_loader(
                    model=model,
                    loader=valid_loader,
                    device=device,
                    loss_fn=loss_fn,
                    midx=midx,
                )
                is_best = stopper.update(val_results["metrics"][monitoring_score])
                if is_best:
                    best_state = copy.deepcopy(model.state_dict())
                logger.info(
                    f"M-{midx} E-{epoch}| Train: {train_loss:.2f} | "
                    f"pred: {total_pred/n_batch:.3f} ps: {total_ps/n_batch:.3f} "
                    f"cs: {total_cs/n_batch:.3f}| Valid: {val_results['loss']:.2f} | "
                    + " | ".join(f"{k}: {v:.3f}" for k, v in val_results["metrics"].items())
                    + f" | best {monitoring_score}: {stopper.best_score:.3f}"
                )
            if stopper.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if test_after_train:
        evaluate_loader(
            model=model,
            loader=test_loader,
            device=device,
            loss_fn=loss_fn,
            midx=midx,
            test_tag=f"Model {midx}",
            print_log=True,
            logger=logger,
        )
    if save_checkpoint and output_dir is not None and best_state is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, str(out / f"checkpoint_{midx}.pt"))

    return {"midx": midx, "model": model}


def evaluate_loader(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn,
    midx: int = 1,
    test_tag: Optional[str] = None,
    print_log: bool = False,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    model.eval()
    yt, yp = torch.Tensor(), torch.Tensor()
    total_loss = 0.0
    steps = 0

    with torch.cuda.device(device):
        with torch.no_grad():
            for batch in loader:
                steps += 1
                loss, _, _, _, yh, y = compute_loss(model, batch, device, loss_fn)
                total_loss += loss.item()
                yp = torch.cat([yp, yh.detach().cpu()], dim=0)
                yt = torch.cat([yt, y.detach().cpu()], dim=0)

    yt_np = yt.view(-1).numpy()
    yp_np = yp.view(-1).numpy()
    results = {
        "midx": midx,
        "y_true": yt_np,
        "y_pred": yp_np,
        "loss": total_loss / max(steps, 1),
        "metrics": evaluation_metrics(yt_np, yp_np, eval_metrics=["mse", "spearman", "pearson"]),
    }
    if print_log and logger and test_tag:
        logger.info(
            f"{test_tag} | Test Loss: {results['loss']:.4f} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in results["metrics"].items())
        )
    return results

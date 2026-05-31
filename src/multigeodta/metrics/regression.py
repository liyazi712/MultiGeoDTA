"""Regression metrics for binding affinity evaluation."""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, List

import numpy as np
from scipy import stats
from sklearn import metrics as sk_metrics


def eval_mse(y_true, y_pred, squared: bool = True) -> float:
    return sk_metrics.mean_squared_error(y_true, y_pred, squared=squared)


def eval_mae(y_true, y_pred) -> float:
    return sk_metrics.mean_absolute_error(y_true, y_pred)


def _as_1d(arr) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def eval_pearson(y_true, y_pred) -> float:
    return stats.pearsonr(_as_1d(y_true), _as_1d(y_pred))[0]


def eval_spearman(y_true, y_pred) -> float:
    return stats.spearmanr(_as_1d(y_true), _as_1d(y_pred))[0]


def eval_r2(y_true, y_pred) -> float:
    return sk_metrics.r2_score(y_true, y_pred)


def eval_auroc(y_true, y_pred) -> float:
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_pred)
    return sk_metrics.auc(fpr, tpr)


def eval_auprc(y_true, y_pred) -> float:
    pre, rec, _ = sk_metrics.precision_recall_curve(y_true, y_pred)
    return sk_metrics.auc(rec, pre)


def r_squared_error(y_obs, y_pred) -> float:
    y_obs = _as_1d(y_obs)
    y_pred = _as_1d(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]
    mult = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = np.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = np.sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred) -> float:
    y_obs = _as_1d(y_obs)
    y_pred = _as_1d(y_pred)
    return np.sum(y_obs * y_pred) / float(np.sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred) -> float:
    y_obs = _as_1d(y_obs)
    y_pred = _as_1d(y_pred)
    k = get_k(y_obs, y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    upp = np.sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = np.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / float(down))


def concordance_index(y, f) -> float:
    y = _as_1d(y)
    f = _as_1d(f)
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    s = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    s = s + 1
                elif u == 0:
                    s = s + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return s / z if z else 0.0


def rm2(y, f) -> float:
    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def evaluation_metrics(
    y_true=None,
    y_pred=None,
    eval_metrics: Iterable[str] = (),
) -> Dict[str, float]:
    results = {}
    for m in eval_metrics:
        if m == "mae":
            s = eval_mae(y_true, y_pred)
        elif m == "mse":
            s = eval_mse(y_true, y_pred, squared=True)
        elif m == "rmse":
            s = eval_mse(y_true, y_pred, squared=False)
        elif m == "pearson":
            s = eval_pearson(y_true, y_pred)
        elif m == "spearman":
            s = eval_spearman(y_true, y_pred)
        elif m == "r2":
            s = eval_r2(y_true, y_pred)
        elif m == "auroc":
            s = eval_auroc(y_true, y_pred)
        elif m == "auprc":
            s = eval_auprc(y_true, y_pred)
        elif m == "ci":
            s = concordance_index(y_true, y_pred)
        elif m == "rm2":
            s = rm2(y_true, y_pred)
        else:
            raise ValueError(f"Unknown evaluation metric: {m}")
        results[m] = s
    return results

"""
Evaluation Metrics for Drug-Target Affinity Prediction

Provides various metrics for assessing model performance:
- Regression metrics: MSE, RMSE, MAE, R², Pearson, Spearman
- Drug discovery metrics: CI (Concordance Index), rm² (modified R²)
- Classification metrics: AUROC, AUPRC
"""

import numpy as np
from scipy import stats
from sklearn import metrics
from typing import Dict, List, Union


def eval_mse(y_true: np.ndarray, y_pred: np.ndarray,
             squared: bool = True) -> float:
    """
    Compute Mean Squared Error (MSE) or Root MSE.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        squared: If True, return MSE; if False, return RMSE

    Returns:
        MSE or RMSE value
    """
    return metrics.mean_squared_error(y_true, y_pred, squared=squared)


def eval_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return eval_mse(y_true, y_pred, squared=False)


def eval_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return metrics.mean_absolute_error(y_true, y_pred)


def eval_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    return stats.pearsonr(y_true, y_pred)[0]


def eval_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient."""
    return stats.spearmanr(y_true, y_pred)[0]


def eval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² (coefficient of determination)."""
    return metrics.r2_score(y_true, y_pred)


def eval_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)


def eval_auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(rec, pre)


def eval_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Concordance Correlation Coefficient (CCC).

    CCC measures agreement between two continuous variables, combining
    precision (Pearson correlation) and accuracy (bias).
    """
    pearson_corr = eval_pearson(y_true, y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    numerator = 2 * pearson_corr * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator if denominator != 0 else 0.0


def _r_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper function for rm2 calculation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    mult = np.sum((y_pred - y_pred_mean) * (y_true - y_true_mean)) ** 2
    y_true_sq = np.sum((y_true - y_true_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)

    return mult / (y_true_sq * y_pred_sq) if (y_true_sq * y_pred_sq) != 0 else 0.0


def _get_k(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper function for rm2 calculation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true * y_pred) / np.sum(y_pred ** 2) if np.sum(y_pred ** 2) != 0 else 0.0


def _squared_error_zero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Helper function for rm2 calculation."""
    k = _get_k(y_true, y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_mean = np.mean(y_true)

    upp = np.sum((y_true - k * y_pred) ** 2)
    down = np.sum((y_true - y_true_mean) ** 2)

    return 1 - (upp / down) if down != 0 else 0.0


def eval_ci(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Concordance Index (CI).

    CI measures the probability that for any two randomly selected samples,
    the sample with higher true value also has higher predicted value.

    Common in drug discovery for ranking compounds.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ind = np.argsort(y_true)
    y_true = y_true[ind]
    y_pred = y_pred[ind]

    i = len(y_true) - 1
    j = i - 1
    z = 0.0
    S = 0.0

    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z += 1
                u = y_pred[i] - y_pred[j]
                if u > 0:
                    S += 1
                elif u == 0:
                    S += 0.5
            j -= 1
        i -= 1
        j = i - 1

    return S / z if z != 0 else 0.0


def eval_rm2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute rm² (modified R²).

    rm² is a stricter metric than R² that penalizes predictions
    that deviate from the ideal y=x line.
    """
    r2 = _r_squared_error(y_true, y_pred)
    r02 = _squared_error_zero(y_true, y_pred)
    return r2 * (1 - np.sqrt(np.abs(r2 * r2 - r02 * r02)))


def evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       eval_metrics: List[str]) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        eval_metrics: List of metric names to compute
            Supported: 'mse', 'rmse', 'mae', 'pearson', 'spearman',
                      'r2', 'auroc', 'auprc', 'ci', 'rm2', 'ccc'

    Returns:
        Dictionary mapping metric names to values

    Example:
        >>> results = evaluation_metrics(y_true, y_pred,
        ...                              ['rmse', 'pearson', 'ci'])
        >>> print(results)
        {'rmse': 0.85, 'pearson': 0.82, 'ci': 0.78}
    """
    metric_funcs = {
        'mse': lambda y, p: eval_mse(y, p, squared=True),
        'rmse': lambda y, p: eval_mse(y, p, squared=False),
        'mae': eval_mae,
        'pearson': eval_pearson,
        'spearman': eval_spearman,
        'r2': eval_r2,
        'auroc': eval_auroc,
        'auprc': eval_auprc,
        'ci': eval_ci,
        'rm2': eval_rm2,
        'ccc': eval_ccc,
    }

    results = {}
    for metric_name in eval_metrics:
        if metric_name not in metric_funcs:
            raise ValueError(f'Unknown evaluation metric: {metric_name}')
        results[metric_name] = metric_funcs[metric_name](y_true, y_pred)

    return results

"""
Early Stopping Utility for MultiGeoDTA
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.

    Monitors a validation metric and stops training when the metric
    stops improving for a specified number of epochs (patience).

    Args:
        patience: Number of epochs to wait before stopping
        eval_freq: Evaluation frequency (epochs between evaluations)
        best_score: Initial best score (optional)
        delta: Minimum change to qualify as improvement
        higher_better: If True, higher scores are better (e.g., accuracy)
                      If False, lower scores are better (e.g., loss)

    Example:
        >>> stopper = EarlyStopping(patience=20, higher_better=False)
        >>> for epoch in range(100):
        ...     val_loss = validate(model)
        ...     is_best = stopper.update(val_loss)
        ...     if is_best:
        ...         save_checkpoint(model)
        ...     if stopper.early_stop:
        ...         print('Early stopping triggered')
        ...         break
    """

    def __init__(self,
                 patience: int = 100,
                 eval_freq: int = 1,
                 best_score: Optional[float] = None,
                 delta: float = 1e-9,
                 higher_better: bool = True):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False

    def not_improved(self, val_score: float) -> bool:
        """
        Check if score has not improved.

        Args:
            val_score: Current validation score

        Returns:
            True if score has not improved
        """
        if np.isnan(val_score):
            return True

        if self.higher_better:
            return val_score < self.best_score + self.delta
        else:
            return val_score > self.best_score - self.delta

    def update(self, val_score: float) -> bool:
        """
        Update early stopping state with new validation score.

        Args:
            val_score: Current validation score

        Returns:
            True if this is a new best score
        """
        if self.best_score is None:
            self.best_score = val_score
            return True

        if self.not_improved(val_score):
            self.counter += self.eval_freq
            if (self.patience is not None) and (self.counter > self.patience):
                self.early_stop = True
            return False
        else:
            self.best_score = val_score
            self.counter = 0
            return True

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        self.best_score = None

    @property
    def should_stop(self) -> bool:
        """Alias for early_stop property."""
        return self.early_stop

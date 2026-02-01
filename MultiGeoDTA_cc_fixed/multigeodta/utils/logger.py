"""
Logging Utility for MultiGeoDTA
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Union


class Logger:
    """
    Logging utility that outputs to both console and file.

    Args:
        logfile: Path to log file (optional)
        level: Logging level (default: INFO)
        name: Logger name (default: root logger)

    Example:
        >>> logger = Logger(logfile='experiment.log')
        >>> logger.info('Training started')
        >>> logger.info('Epoch 1/100, Loss: 0.5')
    """

    def __init__(self, logfile: Optional[Union[str, Path]] = None,
                 level: int = logging.INFO,
                 name: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s\t%(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if logfile is not None:
            logfile = Path(logfile)
            logfile.parent.mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(logfile, 'w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)

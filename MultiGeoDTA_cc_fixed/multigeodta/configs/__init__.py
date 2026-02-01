"""
Configuration Module for MultiGeoDTA

Provides default configurations and configuration management utilities.
"""

from multigeodta.configs.default_config import (
    DefaultConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    get_default_config,
)

__all__ = [
    "DefaultConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "get_default_config",
]

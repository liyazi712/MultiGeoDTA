"""
MultiGeoDTA: Multi-modal Geometric Drug-Target Affinity Prediction

A deep learning framework for predicting drug-target binding affinity
by integrating protein structure, sequence, and molecular features.

Author: Yazi Li
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Yazi Li"
__email__ = "liyazi126@126.com"

from multigeodta.models import DTAModel, ProtGVPModel, DrugGVPModel
from multigeodta.trainer import DTATrainer
from multigeodta.data import DTADataset

__all__ = [
    "DTAModel",
    "ProtGVPModel",
    "DrugGVPModel",
    "DTATrainer",
    "DTADataset",
    "__version__",
]

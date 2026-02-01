"""
Task Definitions for MultiGeoDTA

Provides task-specific data loaders for different DTA benchmarks:
- PDBBind (various versions and splits)
- Virtual Screening (ZINC)
"""

from multigeodta.tasks.pdbbind import (
    PDBBindTask,
    PDBBind2016,
    PDBBind2020,
    PDBBind2021Time,
    PDBBind2021Similarity,
    LPPDBBind,
)
from multigeodta.tasks.virtual_screening import VirtualScreeningTask

__all__ = [
    "PDBBindTask",
    "PDBBind2016",
    "PDBBind2020",
    "PDBBind2021Time",
    "PDBBind2021Similarity",
    "LPPDBBind",
    "VirtualScreeningTask",
]

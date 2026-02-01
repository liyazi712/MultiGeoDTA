"""
PDBBind Task Definitions

Provides data loading and preprocessing for PDBBind benchmark datasets.
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from multigeodta.data.dataset import DTADataset, load_pickle
from multigeodta.data.protein_graph import pdb_to_graphs
from multigeodta.data.mol_graph import sdf_to_graphs
from multigeodta.data.constants import PROTEIN_CHAR, SMILES_CHAR


class PDBBindTask:
    """
    Base class for PDBBind drug-target affinity tasks.

    Handles data loading, graph construction, and dataset preparation
    for training and evaluation.

    Args:
        task_name: Name of the task
        train_csv: Path to training data CSV
        valid_csv: Path to validation data CSV
        test_csv: Path to test data CSV
        train_pdb: Path to training protein structures
        valid_pdb: Path to validation protein structures
        test_pdb: Path to test protein structures
        train_sdf: Path to training molecule structures
        valid_sdf: Path to validation molecule structures
        test_sdf: Path to test molecule structures
        max_seq_len: Maximum protein sequence length
        max_smi_len: Maximum SMILES length
        num_pos_emb: Number of positional embeddings
        num_rbf: Number of RBF kernels
        contact_cutoff: Distance cutoff for protein contacts
        cache_dir: Directory for caching processed data
    """

    def __init__(self,
                 task_name: str,
                 train_csv: Optional[str] = None,
                 valid_csv: Optional[str] = None,
                 test_csv: Optional[str] = None,
                 train_pdb: Optional[str] = None,
                 valid_pdb: Optional[str] = None,
                 test_pdb: Optional[str] = None,
                 train_sdf: Optional[str] = None,
                 valid_sdf: Optional[str] = None,
                 test_sdf: Optional[str] = None,
                 max_seq_len: int = 1024,
                 max_smi_len: int = 128,
                 num_pos_emb: int = 16,
                 num_rbf: int = 16,
                 contact_cutoff: float = 8.0,
                 cache_dir: Optional[str] = None):

        self.task_name = task_name
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

        # Graph construction parameters
        self.prot_params = {
            'num_pos_emb': num_pos_emb,
            'num_rbf': num_rbf,
            'contact_cutoff': contact_cutoff
        }

        # Load data files
        self.train_df = pd.read_csv(train_csv) if train_csv else None
        self.valid_df = pd.read_csv(valid_csv) if valid_csv else None
        self.test_df = pd.read_csv(test_csv) if test_csv else None

        # Load structure data
        self.train_pdb_data = load_pickle(train_pdb) if train_pdb else None
        self.valid_pdb_data = load_pickle(valid_pdb) if valid_pdb else None
        self.test_pdb_data = load_pickle(test_pdb) if test_pdb else None

        self.train_sdf_data = load_pickle(train_sdf) if train_sdf else None
        self.valid_sdf_data = load_pickle(valid_sdf) if valid_sdf else None
        self.test_sdf_data = load_pickle(test_sdf) if test_sdf else None

        # Caching
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Lazy-loaded graph databases
        self._pdb_graph_db = None
        self._sdf_graph_db = None

    def _format_pdb_entry(self, data: Dict) -> Dict:
        """Format PDB entry for graph construction."""
        coords = data["coords"]
        entry = {
            "seq": data["seq"],
            "coords": list(zip(coords["N"], coords["CA"], coords["C"], coords["O"])),
        }
        return entry

    @property
    def pdb_graph_db(self) -> Dict:
        """Lazy-load protein graphs."""
        if self._pdb_graph_db is None:
            pdb_data = {}

            # Combine all PDB data
            for pdb_dict in [self.train_pdb_data, self.valid_pdb_data, self.test_pdb_data]:
                if pdb_dict:
                    pdb_data.update(pdb_dict)

            # Format entries
            formatted = {
                pdb_id: self._format_pdb_entry(data)
                for pdb_id, data in pdb_data.items()
            }

            self._pdb_graph_db = pdb_to_graphs(formatted, self.prot_params)

        return self._pdb_graph_db

    @property
    def sdf_graph_db(self) -> Dict:
        """Lazy-load molecule graphs."""
        if self._sdf_graph_db is None:
            sdf_data = {}

            for sdf_dict in [self.train_sdf_data, self.valid_sdf_data, self.test_sdf_data]:
                if sdf_dict:
                    sdf_data.update(sdf_dict)

            self._sdf_graph_db = sdf_to_graphs(sdf_data)

        return self._sdf_graph_db

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode protein sequence."""
        label = np.zeros(self.max_seq_len)
        for i, aa in enumerate(sequence[:self.max_seq_len]):
            if aa in PROTEIN_CHAR:
                label[i] = PROTEIN_CHAR[aa]
        return label

    def encode_smiles(self, smiles: str) -> np.ndarray:
        """Encode SMILES string."""
        x = np.zeros(self.max_smi_len)
        for i, ch in enumerate(smiles[:self.max_smi_len]):
            x[i] = SMILES_CHAR.get(ch, 66)
        return x

    def encode_pocket(self, sequence: str, positions: List[int]) -> np.ndarray:
        """Encode pocket sequence with masking."""
        pocket = ['<MASK>'] * len(sequence)
        for pos in positions:
            if 0 < pos <= len(sequence):
                pocket[pos - 1] = sequence[pos - 1]
        return self.encode_sequence(''.join(pocket))

    def _build_data_list(self, df: pd.DataFrame, split: str) -> List[Dict]:
        """Build data list from DataFrame."""
        data_list = []

        # Check cache
        if self.cache_dir:
            cache_path = self.cache_dir / f'{self.task_name}_{split}.pkl.gz'
            if cache_path.exists():
                print(f"Loading cached {split} data...")
                return load_pickle(str(cache_path))

        print(f"Building {split} data...")
        for _, entry in df.iterrows():
            pdb_id = entry['PDBname']

            try:
                drug_graph = self.sdf_graph_db[pdb_id]
                prot_graph = self.pdb_graph_db[pdb_id]
            except KeyError:
                print(f"Warning: Missing graph for {pdb_id}, skipping...")
                continue

            sequence = entry['Sequence']
            positions = eval(entry['Position']) if isinstance(entry['Position'], str) else entry['Position']
            smiles = entry['Smile']

            data_list.append({
                'drug_graph': drug_graph,
                'protein_graph': prot_graph,
                'full_sequence': self.encode_sequence(sequence),
                'pocket_sequence': self.encode_pocket(sequence, positions),
                'smile_sequence': self.encode_smiles(smiles),
                'y': entry.get('label', entry.get('affinity', None)),
                'pdb_name': pdb_id,
                'smile': smiles
            })

        # Save cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(str(cache_path), 'wb') as f:
                pickle.dump(data_list, f)
            print(f"Cached {split} data saved.")

        return data_list

    def get_datasets(self) -> Tuple[DTADataset, DTADataset, DTADataset]:
        """
        Get train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        train_data = self._build_data_list(self.train_df, 'train')
        valid_data = self._build_data_list(self.valid_df, 'valid')
        test_data = self._build_data_list(self.test_df, 'test')

        print(f"Dataset sizes - Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")

        return (
            DTADataset(train_data),
            DTADataset(valid_data),
            DTADataset(test_data)
        )


class PDBBind2016(PDBBindTask):
    """PDBBind v2016 benchmark dataset."""

    def __init__(self, data_dir: str = './data/pdbbind_v2016', **kwargs):
        data_dir = Path(data_dir)
        super().__init__(
            task_name='pdbbind_v2016',
            train_csv=str(data_dir / 'last_train_2016.csv'),
            valid_csv=str(data_dir / 'last_valid_2016.csv'),
            test_csv=str(data_dir / 'last_test_2016.csv'),
            train_pdb=str(data_dir / 'pocket_structures_train.pkl.gz'),
            valid_pdb=str(data_dir / 'pocket_structures_valid.pkl.gz'),
            test_pdb=str(data_dir / 'pocket_structures_test.pkl.gz'),
            train_sdf=str(data_dir / 'mol_structures_train.pkl.gz'),
            valid_sdf=str(data_dir / 'mol_structures_valid.pkl.gz'),
            test_sdf=str(data_dir / 'mol_structures_test.pkl.gz'),
            cache_dir=str(data_dir / 'cache'),
            **kwargs
        )


class PDBBind2020(PDBBindTask):
    """PDBBind v2020 benchmark dataset."""

    def __init__(self, data_dir: str = './data/pdbbind_v2020', **kwargs):
        data_dir = Path(data_dir)
        super().__init__(
            task_name='pdbbind_v2020',
            train_csv=str(data_dir / 'last_train_2020.csv'),
            valid_csv=str(data_dir / 'last_valid_2020.csv'),
            test_csv=str(data_dir / 'last_test_2020.csv'),
            train_pdb=str(data_dir / 'pocket_structures_train.pkl.gz'),
            valid_pdb=str(data_dir / 'pocket_structures_valid.pkl.gz'),
            test_pdb=str(data_dir / 'pocket_structures_test.pkl.gz'),
            train_sdf=str(data_dir / 'mol_structures_train.pkl.gz'),
            valid_sdf=str(data_dir / 'mol_structures_valid.pkl.gz'),
            test_sdf=str(data_dir / 'mol_structures_test.pkl.gz'),
            cache_dir=str(data_dir / 'cache'),
            **kwargs
        )


class PDBBind2021Time(PDBBindTask):
    """PDBBind v2021 with time-based split."""

    def __init__(self, data_dir: str = './data/pdbbind_v2021_time', **kwargs):
        data_dir = Path(data_dir)
        super().__init__(
            task_name='pdbbind_v2021_time',
            train_csv=str(data_dir / 'train_2021.csv'),
            valid_csv=str(data_dir / 'valid_2021.csv'),
            test_csv=str(data_dir / 'test_2021.csv'),
            train_pdb=str(data_dir / 'pocket_structures_train.pkl.gz'),
            valid_pdb=str(data_dir / 'pocket_structures_valid.pkl.gz'),
            test_pdb=str(data_dir / 'pocket_structures_test.pkl.gz'),
            train_sdf=str(data_dir / 'mol_structures_train.pkl.gz'),
            valid_sdf=str(data_dir / 'mol_structures_valid.pkl.gz'),
            test_sdf=str(data_dir / 'mol_structures_test.pkl.gz'),
            cache_dir=str(data_dir / 'cache'),
            **kwargs
        )


class PDBBind2021Similarity(PDBBindTask):
    """PDBBind v2021 with similarity-based split."""

    def __init__(self, data_dir: str = './data/pdbbind_v2021_similarity',
                 setting: str = 'new_new', threshold: float = 0.5, **kwargs):
        data_dir = Path(data_dir) / setting
        thre = threshold
        super().__init__(
            task_name='pdbbind_v2021_similarity',
            train_csv=str(data_dir / f'train_{thre}.csv'),
            valid_csv=str(data_dir / f'valid_{thre}.csv'),
            test_csv=str(data_dir / f'test_{thre}.csv'),
            train_pdb=str(data_dir / f'pocket_structures_train_{thre}.pkl.gz'),
            valid_pdb=str(data_dir / f'pocket_structures_valid_{thre}.pkl.gz'),
            test_pdb=str(data_dir / f'pocket_structures_test_{thre}.pkl.gz'),
            train_sdf=str(data_dir / f'mol_structures_train_{thre}.pkl.gz'),
            valid_sdf=str(data_dir / f'mol_structures_valid_{thre}.pkl.gz'),
            test_sdf=str(data_dir / f'mol_structures_test_{thre}.pkl.gz'),
            max_seq_len=800,
            max_smi_len=256,
            cache_dir=str(data_dir / 'cache'),
            **kwargs
        )


class LPPDBBind(PDBBindTask):
    """LP-PDBBind benchmark dataset."""

    def __init__(self, data_dir: str = './data/lp_pdbbind', **kwargs):
        data_dir = Path(data_dir)
        super().__init__(
            task_name='lp_pdbbind',
            train_csv=str(data_dir / 'LP_PDBBind_train.csv'),
            valid_csv=str(data_dir / 'LP_PDBBind_valid.csv'),
            test_csv=str(data_dir / 'LP_PDBBind_test.csv'),
            train_pdb=str(data_dir / 'pocket_structures_train.pkl.gz'),
            valid_pdb=str(data_dir / 'pocket_structures_valid.pkl.gz'),
            test_pdb=str(data_dir / 'pocket_structures_test.pkl.gz'),
            train_sdf=str(data_dir / 'mol_structures_train.pkl.gz'),
            valid_sdf=str(data_dir / 'mol_structures_valid.pkl.gz'),
            test_sdf=str(data_dir / 'mol_structures_test.pkl.gz'),
            cache_dir=str(data_dir / 'cache'),
            **kwargs
        )

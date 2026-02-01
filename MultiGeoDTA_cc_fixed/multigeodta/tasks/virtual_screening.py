"""
Virtual Screening Task

Provides data loading for virtual screening applications where
predictions are made on compounds without known binding affinities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from multigeodta.data.dataset import DTADataset, load_pickle
from multigeodta.data.protein_graph import pdb_to_graphs
from multigeodta.data.mol_graph import sdf_to_graphs
from multigeodta.data.constants import PROTEIN_CHAR, SMILES_CHAR


class VirtualScreeningTask:
    """
    Virtual Screening Task for predicting binding affinities of new compounds.

    This task is designed for inference mode where:
    - A single target protein is used
    - Multiple compounds are screened against this target
    - No ground truth labels are available

    Args:
        compounds_csv: Path to CSV with compound information
        target_pdb: Path to target protein structure
        compounds_sdf: Path to compound structures
        target_sequence: Protein sequence (if not in PDB file)
        pocket_positions: List of pocket residue positions
        max_seq_len: Maximum sequence length
        max_smi_len: Maximum SMILES length
        num_pos_emb: Number of positional embeddings
        num_rbf: Number of RBF kernels
        contact_cutoff: Contact distance cutoff

    Example:
        >>> task = VirtualScreeningTask(
        ...     compounds_csv='zinc_compounds.csv',
        ...     target_pdb='target_pocket.pkl.gz',
        ...     compounds_sdf='zinc_molecules.pkl.gz',
        ...     target_sequence='MVLSPADKTN...',
        ...     pocket_positions=[10, 12, 15, ...]
        ... )
        >>> dataset = task.get_dataset()
        >>> predictions = model.predict(dataset)
    """

    def __init__(self,
                 compounds_csv: str,
                 target_pdb: str,
                 compounds_sdf: str,
                 target_sequence: str,
                 pocket_positions: List[int],
                 target_name: str = 'target',
                 max_seq_len: int = 1024,
                 max_smi_len: int = 128,
                 num_pos_emb: int = 16,
                 num_rbf: int = 16,
                 contact_cutoff: float = 8.0):

        self.compounds_df = pd.read_csv(compounds_csv)
        self.target_pdb_data = load_pickle(target_pdb)
        self.compounds_sdf_data = load_pickle(compounds_sdf)

        self.target_sequence = target_sequence
        self.pocket_positions = pocket_positions
        self.target_name = target_name

        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

        self.prot_params = {
            'num_pos_emb': num_pos_emb,
            'num_rbf': num_rbf,
            'contact_cutoff': contact_cutoff
        }

        self._prot_graph = None
        self._sdf_graph_db = None

    def _format_pdb_entry(self, data: Dict) -> Dict:
        """Format PDB entry for graph construction."""
        coords = data["coords"]
        return {
            "seq": data["seq"],
            "coords": list(zip(coords["N"], coords["CA"], coords["C"], coords["O"])),
        }

    @property
    def protein_graph(self):
        """Get target protein graph."""
        if self._prot_graph is None:
            formatted = {
                self.target_name: self._format_pdb_entry(
                    self.target_pdb_data[self.target_name]
                )
            }
            graphs = pdb_to_graphs(formatted, self.prot_params)
            self._prot_graph = graphs[self.target_name]
        return self._prot_graph

    @property
    def sdf_graph_db(self) -> Dict:
        """Get compound graphs."""
        if self._sdf_graph_db is None:
            self._sdf_graph_db = sdf_to_graphs(self.compounds_sdf_data)
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

    def encode_pocket(self) -> np.ndarray:
        """Encode pocket sequence with masking."""
        pocket = ['<MASK>'] * len(self.target_sequence)
        for pos in self.pocket_positions:
            if 0 < pos <= len(self.target_sequence):
                pocket[pos - 1] = self.target_sequence[pos - 1]
        return self.encode_sequence(''.join(pocket))

    def get_dataset(self) -> DTADataset:
        """
        Build dataset for virtual screening.

        Returns:
            DTADataset for inference
        """
        data_list = []

        # Pre-compute sequence encodings (same for all compounds)
        seq_encoding = self.encode_sequence(self.target_sequence)
        pocket_encoding = self.encode_pocket()

        # Get column names for compound ID and SMILES
        id_col = 'zinc_id' if 'zinc_id' in self.compounds_df.columns else 'compound_id'
        smi_col = 'SMILES' if 'SMILES' in self.compounds_df.columns else 'smiles'

        for _, entry in self.compounds_df.iterrows():
            compound_id = entry[id_col]

            try:
                drug_graph = self.sdf_graph_db[compound_id]
            except KeyError:
                print(f"Warning: Missing graph for {compound_id}, skipping...")
                continue

            smiles = entry[smi_col]

            data_list.append({
                'drug_graph': drug_graph,
                'protein_graph': self.protein_graph,
                'full_sequence': seq_encoding.copy(),
                'pocket_sequence': pocket_encoding.copy(),
                'smile_sequence': self.encode_smiles(smiles),
                'y': None,  # No labels for virtual screening
                'pdb_name': compound_id,
                'smile': smiles
            })

        print(f"Built dataset with {len(data_list)} compounds")
        return DTADataset(data_list)

    def get_compound_ids(self) -> List[str]:
        """Get list of compound IDs."""
        id_col = 'zinc_id' if 'zinc_id' in self.compounds_df.columns else 'compound_id'
        return self.compounds_df[id_col].tolist()

    def create_results_dataframe(self, predictions: np.ndarray) -> pd.DataFrame:
        """
        Create results DataFrame from predictions.

        Args:
            predictions: Array of predicted binding affinities

        Returns:
            DataFrame with compound IDs, SMILES, and predictions
        """
        id_col = 'zinc_id' if 'zinc_id' in self.compounds_df.columns else 'compound_id'
        smi_col = 'SMILES' if 'SMILES' in self.compounds_df.columns else 'smiles'

        results = pd.DataFrame({
            'compound_id': self.compounds_df[id_col],
            'smiles': self.compounds_df[smi_col],
            'predicted_affinity': predictions
        })

        # Sort by predicted affinity (lower is better for pKd)
        results = results.sort_values('predicted_affinity', ascending=False)
        results['rank'] = range(1, len(results) + 1)

        return results

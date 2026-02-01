"""
Dataset Classes for Drug-Target Affinity Prediction

Provides PyTorch-compatible dataset and dataloader for DTA tasks.
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch_geometric.data import Batch
from typing import Dict, List, Optional, Tuple, Any

from multigeodta.data.constants import PROTEIN_CHAR, SMILES_CHAR


def load_pickle(input_path: str) -> Any:
    """Load gzip-compressed pickle file."""
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, output_path: str):
    """Save object to gzip-compressed pickle file."""
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


class DTADataset(data.Dataset):
    """
    Drug-Target Affinity Dataset.

    A PyTorch Dataset for loading and processing drug-target binding affinity data.

    Args:
        data_list: List of data dictionaries containing drug/protein graphs and labels

    Each data entry should contain:
        - drug_graph: PyG Data object for drug molecule
        - protein_graph: PyG Data object for protein structure
        - full_sequence: Encoded protein sequence
        - pocket_sequence: Encoded pocket sequence
        - smile_sequence: Encoded SMILES sequence
        - y: Binding affinity label (optional for inference)
        - smile: Original SMILES string

    Example:
        >>> dataset = DTADataset(data_list)
        >>> drug, prot, seq, poc_seq, smi_seq, label, smile = dataset[0]
    """

    def __init__(self, data_list: List[Dict]):
        super(DTADataset, self).__init__()
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        entry = self.data_list[idx]

        drug = entry['drug_graph']
        prot = entry['protein_graph']
        full_seq = entry['full_sequence']
        poc_seq = entry['pocket_sequence']
        smile_encode = entry['smile_sequence']
        smile = entry['smile']

        if 'y' in entry and entry['y'] is not None:
            y = entry['y']
            return drug, prot, full_seq, poc_seq, smile_encode, float(y), smile
        else:
            return drug, prot, full_seq, poc_seq, smile_encode, smile

    @staticmethod
    def collate_fn(sample: List) -> Dict[str, Any]:
        """
        Collate function for batching samples.

        Args:
            sample: List of samples from __getitem__

        Returns:
            Dictionary containing batched data
        """
        if len(sample[0]) == 7:
            # With labels
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, label, smile = \
                map(list, zip(*sample))

            compound_graph = Batch.from_data_list(compound_graph)
            protein_graph = Batch.from_data_list(protein_graph)
            full_seq = torch.tensor(np.array(full_seq)).long()
            poc_seq = torch.tensor(np.array(poc_seq)).long()
            smile_seq = torch.tensor(np.array(smile_seq)).long()
            label = torch.FloatTensor(label)

            return {
                'drug': compound_graph,
                'protein': protein_graph,
                'full_seq': full_seq,
                'poc_seq': poc_seq,
                'smile_seq': smile_seq,
                'y': label,
                'SMILES': list(smile)
            }

        elif len(sample[0]) == 6:
            # Without labels (inference mode)
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, smile = \
                map(list, zip(*sample))

            compound_graph = Batch.from_data_list(compound_graph)
            protein_graph = Batch.from_data_list(protein_graph)
            full_seq = torch.tensor(np.array(full_seq)).long()
            poc_seq = torch.tensor(np.array(poc_seq)).long()
            smile_seq = torch.tensor(np.array(smile_seq)).long()

            return {
                'drug': compound_graph,
                'protein': protein_graph,
                'full_seq': full_seq,
                'poc_seq': poc_seq,
                'smile_seq': smile_seq,
                'SMILES': list(smile)
            }
        else:
            raise ValueError(f"Unexpected sample length: {len(sample[0])}")

    def collate(self, sample):
        """Backward compatible collate method."""
        return self.collate_fn(sample)


class DTADataLoader(data.DataLoader):
    """
    DataLoader for DTA datasets with automatic collate function.

    Args:
        dataset: DTADataset instance
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        **kwargs: Additional arguments to torch.utils.data.DataLoader
    """

    def __init__(self, dataset: DTADataset, batch_size: int = 32,
                 shuffle: bool = False, num_workers: int = 0, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=DTADataset.collate_fn,
            **kwargs
        )


class SequenceEncoder:
    """
    Utility class for encoding sequences.

    Args:
        max_seq_len: Maximum protein sequence length
        max_smi_len: Maximum SMILES sequence length
    """

    def __init__(self, max_seq_len: int = 1024, max_smi_len: int = 128):
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

    def encode_protein_sequence(self, sequence: str) -> np.ndarray:
        """
        Encode protein sequence to numerical array.

        Args:
            sequence: Protein sequence string

        Returns:
            Encoded sequence array of shape (max_seq_len,)
        """
        label = np.zeros(self.max_seq_len)
        for i, aa in enumerate(sequence[:self.max_seq_len]):
            if aa in PROTEIN_CHAR:
                label[i] = PROTEIN_CHAR[aa]
        return label

    def encode_smiles(self, smiles: str) -> np.ndarray:
        """
        Encode SMILES string to numerical array.

        Args:
            smiles: SMILES string

        Returns:
            Encoded SMILES array of shape (max_smi_len,)
        """
        x = np.zeros(self.max_smi_len)
        for i, ch in enumerate(smiles[:self.max_smi_len]):
            x[i] = SMILES_CHAR.get(ch, 66)  # Default unknown char to 66
        return x

    def encode_pocket_sequence(self, sequence: str, positions: List[int]) -> np.ndarray:
        """
        Encode pocket sequence with positional masking.

        Args:
            sequence: Full protein sequence
            positions: List of pocket residue positions (1-indexed)

        Returns:
            Encoded pocket sequence with non-pocket residues masked
        """
        pocket_seq = ['<MASK>'] * len(sequence)
        for pos in positions:
            if 0 < pos <= len(sequence):
                pocket_seq[pos - 1] = sequence[pos - 1]
        return self.encode_protein_sequence(''.join(pocket_seq))

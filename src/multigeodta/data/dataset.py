"""PyTorch Dataset and collate for drug-target affinity."""

from __future__ import annotations

import numpy as np
import torch
import torch.utils.data as data
from torch_geometric.data import Batch


class DTADataset(data.Dataset):
    """Drug-target binding affinity dataset with optional labels."""

    def __init__(self, data_list=None):
        super().__init__()
        self.data_list = data_list or []

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        entry = self.data_list[idx]
        drug = entry["drug_graph"]
        prot = entry["protein_graph"]
        full_seq = entry["full_sequence"]
        poc_seq = entry["pocket_sequence"]
        smile_encode = entry["smile_sequence"]
        smile = entry["smile"]
        if entry.get("y") is not None:
            return drug, prot, full_seq, poc_seq, smile_encode, float(entry["y"]), smile
        return drug, prot, full_seq, poc_seq, smile_encode, smile

    def collate(self, sample):
        n_fields = len(sample[0])
        if n_fields == 7:
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, label, smile = zip(*sample)
            label = torch.FloatTensor(label)
            item = {
                "drug": Batch.from_data_list(list(compound_graph)),
                "protein": Batch.from_data_list(list(protein_graph)),
                "full_seq": torch.tensor(np.array(full_seq)).long(),
                "poc_seq": torch.tensor(np.array(poc_seq)).long(),
                "smile_seq": torch.tensor(np.array(smile_seq)).long(),
                "y": label,
                "SMILES": list(smile),
            }
            return item
        if n_fields == 6:
            compound_graph, protein_graph, full_seq, poc_seq, smile_seq, smile = zip(*sample)
            return {
                "drug": Batch.from_data_list(list(compound_graph)),
                "protein": Batch.from_data_list(list(protein_graph)),
                "full_seq": torch.tensor(np.array(full_seq)).long(),
                "poc_seq": torch.tensor(np.array(poc_seq)).long(),
                "smile_seq": torch.tensor(np.array(smile_seq)).long(),
                "SMILES": list(smile),
            }
        raise ValueError(f"Expected 6 or 7 fields per sample, got {n_fields}")

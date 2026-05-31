"""Base task: load splits, featurize graphs, build DTADataset."""

from __future__ import annotations

import ast
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from multigeodta.data import featurizers
from multigeodta.data.dataset import DTADataset
from multigeodta.data.vocab import PROTEIN_CHAR, SMILES_CHAR_SET, SMILES_DEFAULT_IDX
from multigeodta.utils.paths import task_data_dir


def load_dict(path: Union[str, Path]) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def parse_positions(value: Any) -> List[int]:
    """Parse pocket residue indices from CSV (JSON list or Python literal)."""
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return [int(x) for x in parsed]
    raise TypeError(f"Cannot parse positions from {type(value)}")


class DTATask:
    """Drug-target affinity benchmark task."""

    task_name: str = "base"

    def __init__(
        self,
        task_name: str,
        train_data: Optional[pd.DataFrame] = None,
        valid_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        train_pdb_data: Optional[dict] = None,
        valid_pdb_data: Optional[dict] = None,
        test_pdb_data: Optional[dict] = None,
        train_sdf_data: Optional[dict] = None,
        valid_sdf_data: Optional[dict] = None,
        test_sdf_data: Optional[dict] = None,
        max_seq_len: int = 1024,
        max_smi_len: int = 128,
        num_pos_emb: int = 16,
        num_rbf: int = 16,
        contact_cutoff: float = 8.0,
        data_root: Optional[Path] = None,
        inference_target: Optional[dict] = None,
    ):
        self.task_name = task_name
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_pdb_data = train_pdb_data
        self.valid_pdb_data = valid_pdb_data
        self.test_pdb_data = test_pdb_data
        self.train_sdf_data = train_sdf_data
        self.valid_sdf_data = valid_sdf_data
        self.test_sdf_data = test_sdf_data
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.prot_featurize_params = dict(
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self.data_root = Path(data_root) if data_root else task_data_dir(task_name)
        self.inference_target = inference_target or {}
        self._pdb_graph_db = None
        self._drug_sdf_db = None

    def _data_path(self, *parts: str) -> Path:
        return self.data_root.joinpath(*parts)

    def _processed_cache(self, split: str) -> Path:
        return self._data_path(f"processed_data_dict_{split}.pkl.gz")

    def _format_pdb_entry(self, _data: dict) -> dict:
        coords = _data["coords"]
        entry = {
            "seq": _data["seq"],
            "coords": list(zip(coords["N"], coords["CA"], coords["C"], coords["O"])),
        }
        return entry

    @property
    def pdb_graph_db(self) -> dict:
        if self._pdb_graph_db is None:
            if self.test_pdb_data is not None and self.valid_pdb_data is not None:
                combined = {**self.train_pdb_data, **self.valid_pdb_data, **self.test_pdb_data}
            else:
                combined = {**self.test_pdb_data}
            pdbid_entry = {
                k: self._format_pdb_entry(v) for k, v in combined.items()
            }
            self._pdb_graph_db = featurizers.pdb_to_graphs(
                pdbid_entry, self.prot_featurize_params
            )
        return self._pdb_graph_db

    @property
    def drug_sdf_db(self) -> dict:
        if self._drug_sdf_db is None:
            if self.train_sdf_data is not None and self.valid_sdf_data is not None:
                combined = {
                    **self.train_sdf_data,
                    **self.valid_sdf_data,
                    **self.test_sdf_data,
                }
            else:
                combined = {**self.test_sdf_data}
            self._drug_sdf_db = featurizers.sdf_to_graphs(combined)
        return self._drug_sdf_db

    def encode_protein(self, line: str) -> np.ndarray:
        label = np.zeros(self.max_seq_len)
        for i, lab in enumerate(line[: self.max_seq_len]):
            label[i] = PROTEIN_CHAR.get(lab, PROTEIN_CHAR["<MASK>"])
        return label

    def encode_smiles(self, smiles: str) -> np.ndarray:
        x = np.zeros(self.max_smi_len)
        for i, ch in enumerate(smiles[: self.max_smi_len]):
            x[i] = SMILES_CHAR_SET.get(ch, SMILES_DEFAULT_IDX)
        return x

    @staticmethod
    def mask_pocket(seq: str, position: Sequence[int]) -> List[str]:
        res = ["<MASK>"] * len(seq)
        for i in position:
            res[i - 1] = seq[i - 1]
        return res

    def _build_labeled_split(
        self, frame: pd.DataFrame, split: str, save_processed: bool
    ) -> DTADataset:
        cache = self._processed_cache(split)
        if cache.exists():
            with gzip.open(cache, "rb") as f:
                data_list = pickle.load(f)
            print(f"{split} data loaded from cache: {cache}")
            return DTADataset(data_list)

        print(f"Processing {split} split from scratch...")
        data_list = []
        for entry in frame.to_dict("records"):
            pdb_id = entry["PDBname"]
            df = self.drug_sdf_db[pdb_id]
            pf = self.pdb_graph_db[pdb_id]
            seq = entry["Sequence"]
            position = parse_positions(entry["Position"])
            smile = entry["Smile"]
            data_list.append(
                {
                    "drug_graph": df,
                    "protein_graph": pf,
                    "full_sequence": self.encode_protein(seq),
                    "pocket_sequence": self.encode_protein(self.mask_pocket(seq, position)),
                    "smile_sequence": self.encode_smiles(smile),
                    "y": entry["label"],
                    "pdb_name": pdb_id,
                    "smile": smile,
                }
            )
        if save_processed:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(cache, "wb") as f:
                pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"{split} data saved to {cache}")
        return DTADataset(data_list)

    def _build_inference_split(
        self, frame: pd.DataFrame, save_processed: bool
    ) -> DTADataset:
        cache = self._data_path("processed_data_dict_new.pkl.gz")
        if cache.exists():
            with gzip.open(cache, "rb") as f:
                data_list = pickle.load(f)
            print(f"Inference data loaded from cache: {cache}")
            return DTADataset(data_list)

        target = self.inference_target
        seq = target.get("protein_sequence", "")
        position = target.get("pocket_positions", [])
        if not seq or not position:
            raise ValueError(
                "Virtual screening requires inference_target with "
                "protein_sequence and pocket_positions (see configs/tasks/zinc_vs.yaml)"
            )

        print("Processing inference split from scratch...")
        data_list = []
        for entry in frame.to_dict("records"):
            pdb_id = entry["zinc_id"]
            pf = self.pdb_graph_db["target"]
            df = self.drug_sdf_db[pdb_id]
            smile = entry["SMILES"]
            pocket = self.mask_pocket(seq, position)
            data_list.append(
                {
                    "drug_graph": df,
                    "protein_graph": pf,
                    "full_sequence": self.encode_protein(seq),
                    "pocket_sequence": self.encode_protein(pocket),
                    "smile_sequence": self.encode_smiles(smile),
                    "pdb_name": pdb_id,
                    "smile": smile,
                }
            )
        if save_processed:
            with gzip.open(cache, "wb") as f:
                pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        return DTADataset(data_list)

    def get_split(self):
        split_df = {
            "train": self.train_data,
            "valid": self.valid_data,
            "test": self.test_data,
        }
        split_data = {}
        if self.train_data is not None and self.valid_data is not None:
            split_data["train"] = self._build_labeled_split(
                self.train_data, "train", save_processed=True
            )
            split_data["valid"] = self._build_labeled_split(
                self.valid_data, "valid", save_processed=True
            )
            split_data["test"] = self._build_labeled_split(
                self.test_data, "test", save_processed=True
            )
            print(
                "train:", len(split_data["train"]),
                "valid:", len(split_data["valid"]),
                "test:", len(split_data["test"]),
            )
        else:
            split_data["test"] = self._build_inference_split(
                self.test_data, save_processed=True
            )
            print("test only:", len(split_data["test"]))
        return split_data, split_df

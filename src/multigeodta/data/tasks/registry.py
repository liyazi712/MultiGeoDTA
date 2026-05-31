"""Task registry and factory for all benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

import pandas as pd

from multigeodta.data.tasks.base import DTATask, load_dict
from multigeodta.utils.paths import get_data_dir, task_data_dir

SMOKE_N_SAMPLES = 8


def _subset_frames(train, valid, test, n: int = SMOKE_N_SAMPLES):
    return train.head(n), valid.head(n), test.head(n)


def _subset_pkl(pkl_dict: dict, pdb_names) -> dict:
    names = set(pdb_names)
    return {k: v for k, v in pkl_dict.items() if k in names}


def _load_pdbbind_split(
    task_name: str,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    pdb_prefix: str,
    sdf_prefix: str,
    max_seq_len: int = 1024,
    max_smi_len: int = 128,
    num_pos_emb: int = 16,
    num_rbf: int = 16,
    contact_cutoff: float = 8.0,
    data_root: Optional[Path] = None,
    lp_index_col: bool = False,
) -> DTATask:
    root = data_root or task_data_dir(task_name)

    def _read_csv(name: str) -> pd.DataFrame:
        path = root / name
        if lp_index_col:
            return pd.read_csv(path, index_col=0)
        return pd.read_csv(path)

    return DTATask(
        task_name=task_name,
        train_data=_read_csv(train_csv),
        valid_data=_read_csv(valid_csv),
        test_data=_read_csv(test_csv),
        train_pdb_data=load_dict(root / f"{pdb_prefix}_train.pkl.gz"),
        valid_pdb_data=load_dict(root / f"{pdb_prefix}_valid.pkl.gz"),
        test_pdb_data=load_dict(root / f"{pdb_prefix}_test.pkl.gz"),
        train_sdf_data=load_dict(root / f"{sdf_prefix}_train.pkl.gz"),
        valid_sdf_data=load_dict(root / f"{sdf_prefix}_valid.pkl.gz"),
        test_sdf_data=load_dict(root / f"{sdf_prefix}_test.pkl.gz"),
        max_seq_len=max_seq_len,
        max_smi_len=max_smi_len,
        num_pos_emb=num_pos_emb,
        num_rbf=num_rbf,
        contact_cutoff=contact_cutoff,
        data_root=root,
    )


ROBUSTNESS_VARIANTS = frozenset(
    {
        "noised_scale_0.2",
        "noised_scale_0.4",
        "noised_scale_0.6",
        "noised_scale_0.8",
        "noised_scale_1.0",
        "missing_0.2",
        "missing_0.4",
        "missing_0.6",
        "missing_0.8",
    }
)


class PDBbindV2016RobustnessTask(DTATask):
    """PDBBind v2016 robustness benchmark: perturbed train, standard valid/test."""

    def __init__(self, variant: str = "noised_scale_0.2", **kwargs):
        if variant not in ROBUSTNESS_VARIANTS:
            raise ValueError(
                f"Unknown robustness variant '{variant}'. "
                f"Choose from: {sorted(ROBUSTNESS_VARIANTS)}"
            )
        self.variant = variant
        root = kwargs.get("data_root") or task_data_dir("pdbbind_v2016")
        rob_dir = root / "pdbbind_v2016_robustness_test"
        train_data = pd.read_csv(rob_dir / f"last_train_2016_{variant}.csv")
        valid_data = pd.read_csv(root / "last_valid_2016.csv")
        test_data = pd.read_csv(root / "last_test_2016.csv")
        train_pdb = load_dict(root / "pocket_structures_train.pkl.gz")
        valid_pdb = load_dict(root / "pocket_structures_valid.pkl.gz")
        test_pdb = load_dict(root / "pocket_structures_test.pkl.gz")
        train_sdf = load_dict(root / "mol_structures_train.pkl.gz")
        valid_sdf = load_dict(root / "mol_structures_valid.pkl.gz")
        test_sdf = load_dict(root / "mol_structures_test.pkl.gz")
        if kwargs.get("smoke"):
            train_data, valid_data, test_data = _subset_frames(
                train_data, valid_data, test_data
            )
            train_pdb = _subset_pkl(train_pdb, train_data["PDBname"])
            valid_pdb = _subset_pkl(valid_pdb, valid_data["PDBname"])
            test_pdb = _subset_pkl(test_pdb, test_data["PDBname"])
            train_sdf = _subset_pkl(train_sdf, train_data["PDBname"])
            valid_sdf = _subset_pkl(valid_sdf, valid_data["PDBname"])
            test_sdf = _subset_pkl(test_sdf, test_data["PDBname"])
        super().__init__(
            task_name="pdbbind_v2016_robustness",
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            train_pdb_data=train_pdb,
            valid_pdb_data=valid_pdb,
            test_pdb_data=test_pdb,
            train_sdf_data=train_sdf,
            valid_sdf_data=valid_sdf,
            test_sdf_data=test_sdf,
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=root,
        )

    def _processed_cache(self, split: str) -> Path:
        if split == "train":
            return self._data_path(f"processed_data_dict_train_{self.variant}.pkl.gz")
        return self._data_path(f"processed_data_dict_{split}.pkl.gz")


class PDBbindV2016Task(DTATask):
    def __init__(self, **kwargs):
        kwargs.setdefault("data_root", task_data_dir("pdbbind_v2016"))
        root = kwargs["data_root"]
        train_data = pd.read_csv(root / "last_train_2016.csv")
        valid_data = pd.read_csv(root / "last_valid_2016.csv")
        test_data = pd.read_csv(root / "last_test_2016.csv")
        train_pdb = load_dict(root / "pocket_structures_train.pkl.gz")
        valid_pdb = load_dict(root / "pocket_structures_valid.pkl.gz")
        test_pdb = load_dict(root / "pocket_structures_test.pkl.gz")
        train_sdf = load_dict(root / "mol_structures_train.pkl.gz")
        valid_sdf = load_dict(root / "mol_structures_valid.pkl.gz")
        test_sdf = load_dict(root / "mol_structures_test.pkl.gz")
        if kwargs.get("smoke"):
            train_data, valid_data, test_data = _subset_frames(train_data, valid_data, test_data)
            train_pdb = _subset_pkl(train_pdb, train_data["PDBname"])
            valid_pdb = _subset_pkl(valid_pdb, valid_data["PDBname"])
            test_pdb = _subset_pkl(test_pdb, test_data["PDBname"])
            train_sdf = _subset_pkl(train_sdf, train_data["PDBname"])
            valid_sdf = _subset_pkl(valid_sdf, valid_data["PDBname"])
            test_sdf = _subset_pkl(test_sdf, test_data["PDBname"])
        super().__init__(
            task_name="pdbbind_v2016",
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            train_pdb_data=train_pdb,
            valid_pdb_data=valid_pdb,
            test_pdb_data=test_pdb,
            train_sdf_data=train_sdf,
            valid_sdf_data=valid_sdf,
            test_sdf_data=test_sdf,
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=kwargs["data_root"],
        )


class PDBbindV2020Task(PDBbindV2016Task):
    def __init__(self, **kwargs):
        kwargs.setdefault("data_root", task_data_dir("pdbbind_v2020"))
        DTATask.__init__(
            self,
            task_name="pdbbind_v2020",
            train_data=pd.read_csv(kwargs["data_root"] / "last_train_2020.csv"),
            valid_data=pd.read_csv(kwargs["data_root"] / "last_valid_2020.csv"),
            test_data=pd.read_csv(kwargs["data_root"] / "last_test_2020.csv"),
            train_pdb_data=load_dict(kwargs["data_root"] / "pocket_structures_train.pkl.gz"),
            valid_pdb_data=load_dict(kwargs["data_root"] / "pocket_structures_valid.pkl.gz"),
            test_pdb_data=load_dict(kwargs["data_root"] / "pocket_structures_test.pkl.gz"),
            train_sdf_data=load_dict(kwargs["data_root"] / "mol_structures_train.pkl.gz"),
            valid_sdf_data=load_dict(kwargs["data_root"] / "mol_structures_valid.pkl.gz"),
            test_sdf_data=load_dict(kwargs["data_root"] / "mol_structures_test.pkl.gz"),
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=kwargs["data_root"],
        )


class PDBbindV2021TimeTask(PDBbindV2016Task):
    def __init__(self, **kwargs):
        root = kwargs.get("data_root") or task_data_dir("pdbbind_v2021_time")
        DTATask.__init__(
            self,
            task_name="pdbbind_v2021_time",
            train_data=pd.read_csv(root / "train_2021.csv"),
            valid_data=pd.read_csv(root / "valid_2021.csv"),
            test_data=pd.read_csv(root / "test_2021.csv"),
            train_pdb_data=load_dict(root / "pocket_structures_train.pkl.gz"),
            valid_pdb_data=load_dict(root / "pocket_structures_valid.pkl.gz"),
            test_pdb_data=load_dict(root / "pocket_structures_test.pkl.gz"),
            train_sdf_data=load_dict(root / "mol_structures_train.pkl.gz"),
            valid_sdf_data=load_dict(root / "mol_structures_valid.pkl.gz"),
            test_sdf_data=load_dict(root / "mol_structures_test.pkl.gz"),
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=root,
        )


class PDBbindV2021SimilarityTask(DTATask):
    def __init__(
        self,
        setting: str = "new_new",
        thre: str = "0.5",
        **kwargs,
    ):
        root = kwargs.get("data_root") or task_data_dir("pdbbind_v2021_similarity")
        sub = root / setting
        DTATask.__init__(
            self,
            task_name="pdbbind_v2021_similarity",
            train_data=pd.read_csv(sub / f"train_{thre}.csv"),
            valid_data=pd.read_csv(sub / f"valid_{thre}.csv"),
            test_data=pd.read_csv(sub / f"test_{thre}.csv"),
            train_pdb_data=load_dict(sub / f"pocket_structures_train_{thre}.pkl.gz"),
            valid_pdb_data=load_dict(sub / f"pocket_structures_valid_{thre}.pkl.gz"),
            test_pdb_data=load_dict(sub / f"pocket_structures_test_{thre}.pkl.gz"),
            train_sdf_data=load_dict(sub / f"mol_structures_train_{thre}.pkl.gz"),
            valid_sdf_data=load_dict(sub / f"mol_structures_valid_{thre}.pkl.gz"),
            test_sdf_data=load_dict(sub / f"mol_structures_test_{thre}.pkl.gz"),
            max_seq_len=kwargs.get("max_seq_len", 800),
            max_smi_len=kwargs.get("max_smi_len", 256),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=root,
        )
        self.setting = setting
        self.thre = thre


class LPPDBbindTask(DTATask):
    def __init__(self, **kwargs):
        root = kwargs.get("data_root") or task_data_dir("lp_pdbbind")
        DTATask.__init__(
            self,
            task_name="lp_pdbbind",
            train_data=pd.read_csv(root / "LP_PDBBind_train.csv", index_col=0),
            valid_data=pd.read_csv(root / "LP_PDBBind_valid.csv", index_col=0),
            test_data=pd.read_csv(root / "LP_PDBBind_test.csv", index_col=0),
            train_pdb_data=load_dict(root / "pocket_structures_train.pkl.gz"),
            valid_pdb_data=load_dict(root / "pocket_structures_valid.pkl.gz"),
            test_pdb_data=load_dict(root / "pocket_structures_test.pkl.gz"),
            train_sdf_data=load_dict(root / "mol_structures_train.pkl.gz"),
            valid_sdf_data=load_dict(root / "mol_structures_valid.pkl.gz"),
            test_sdf_data=load_dict(root / "mol_structures_test.pkl.gz"),
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=root,
        )


class ZincVirtualScreenTask(DTATask):
    def __init__(self, inference_target: Optional[dict] = None, **kwargs):
        root = kwargs.get("data_root") or task_data_dir("zinc")
        DTATask.__init__(
            self,
            task_name="zinc",
            test_data=pd.read_csv(root / "processed_zinc_CB1R.csv"),
            test_pdb_data=load_dict(root / "pocket_structures.pkl.gz"),
            test_sdf_data=load_dict(root / "mol_structures.pkl.gz"),
            max_seq_len=kwargs.get("max_seq_len", 1024),
            max_smi_len=kwargs.get("max_smi_len", 128),
            num_pos_emb=kwargs.get("num_pos_emb", 16),
            num_rbf=kwargs.get("num_rbf", 16),
            contact_cutoff=kwargs.get("contact_cutoff", 8.0),
            data_root=root,
            inference_target=inference_target,
        )


TASK_REGISTRY: Dict[str, Type[DTATask]] = {
    "pdbbind_v2016": PDBbindV2016Task,
    "pdbbind_v2016_robustness": PDBbindV2016RobustnessTask,
    "pdbbind_v2020": PDBbindV2020Task,
    "pdbbind_v2021_time": PDBbindV2021TimeTask,
    "pdbbind_v2021_similarity": PDBbindV2021SimilarityTask,
    "lp_pdbbind": LPPDBbindTask,
    "zinc": ZincVirtualScreenTask,
}


def build_task(task_name: str, config: Optional[Dict[str, Any]] = None) -> DTATask:
    """Instantiate a registered task from config dict."""
    config = dict(config or {})
    if task_name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    cls = TASK_REGISTRY[task_name]
    kwargs = {
        "num_pos_emb": config.get("num_pos_emb", 16),
        "num_rbf": config.get("num_rbf", 16),
        "contact_cutoff": config.get("contact_cutoff", 8.0),
    }
    if config.get("data_dir"):
        kwargs["data_root"] = Path(config["data_dir"])
    elif config.get("data_root"):
        kwargs["data_root"] = Path(config["data_root"])
    elif task_name == "pdbbind_v2016_robustness":
        kwargs["data_root"] = task_data_dir("pdbbind_v2016")
    else:
        kwargs["data_root"] = task_data_dir(task_name)

    if config.get("smoke"):
        kwargs["smoke"] = True
    if task_name == "pdbbind_v2016_robustness":
        kwargs["variant"] = config.get("variant", "noised_scale_0.2")
    if task_name == "pdbbind_v2021_similarity":
        kwargs["setting"] = config.get("split_method", config.get("setting", "new_new"))
        kwargs["thre"] = str(config.get("thre", "0.5"))
    if task_name == "zinc":
        kwargs["inference_target"] = config.get("target") or config.get("inference_target")

    return cls(**kwargs)

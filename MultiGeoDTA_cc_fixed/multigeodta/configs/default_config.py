"""
Default Configuration for MultiGeoDTA

Provides dataclass-based configuration management.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any
import yaml


@dataclass
class ModelConfig:
    """Configuration for DTAModel architecture."""

    # Drug GVP parameters
    drug_node_in_dim: Tuple[int, int] = (86, 1)
    drug_node_h_dims: Tuple[int, int] = (128, 64)
    drug_edge_in_dim: Tuple[int, int] = (24, 3)
    drug_edge_h_dims: Tuple[int, int] = (32, 1)

    # Protein GVP parameters
    prot_node_in_dim: Tuple[int, int] = (6, 3)
    prot_node_h_dims: Tuple[int, int] = (128, 64)
    prot_edge_in_dim: Tuple[int, int] = (32, 1)
    prot_edge_h_dims: Tuple[int, int] = (32, 1)

    # MLP parameters
    mlp_dims: List[int] = field(default_factory=lambda: [1024, 512])
    mlp_dropout: float = 0.25

    # Sequence encoder parameters
    seq_embedding_dim: int = 256


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    n_ensembles: int = 5
    n_epochs: int = 100
    batch_size: int = 128
    lr: float = 0.0001
    patience: int = 20
    eval_freq: int = 1

    # Monitoring
    monitor_metric: str = 'mse'

    # Device
    device: int = 0
    parallel: bool = False

    # Saving
    output_dir: str = './output'
    save_log: bool = True
    save_checkpoint: bool = True
    save_prediction: bool = True


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Graph construction
    contact_cutoff: float = 8.0
    num_pos_emb: int = 16
    num_rbf: int = 16

    # Sequence lengths
    max_seq_len: int = 1024
    max_smi_len: int = 128

    # Data paths (to be set by user)
    train_csv: Optional[str] = None
    valid_csv: Optional[str] = None
    test_csv: Optional[str] = None
    train_pdb: Optional[str] = None
    valid_pdb: Optional[str] = None
    test_pdb: Optional[str] = None
    train_sdf: Optional[str] = None
    valid_sdf: Optional[str] = None
    test_sdf: Optional[str] = None


@dataclass
class DefaultConfig:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Task configuration
    task: str = 'pdbbind_v2016'
    split_method: str = 'random'
    threshold: float = 0.5

    # Random seed
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> 'DefaultConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            task=config_dict.get('task', 'pdbbind_v2016'),
            split_method=config_dict.get('split_method', 'random'),
            threshold=config_dict.get('threshold', 0.5),
            seed=config_dict.get('seed', 42)
        )


def get_default_config(task: str = 'pdbbind_v2016') -> DefaultConfig:
    """
    Get default configuration for a specific task.

    Args:
        task: Task name (pdbbind_v2016, pdbbind_v2020, pdbbind_v2021_time,
              pdbbind_v2021_similarity, lp_pdbbind, zinc)

    Returns:
        DefaultConfig instance
    """
    config = DefaultConfig(task=task)

    # Task-specific adjustments
    if task == 'pdbbind_v2021_similarity':
        config.data.max_seq_len = 800
        config.data.max_smi_len = 256

    return config


# Predefined configurations for common tasks
TASK_CONFIGS = {
    'pdbbind_v2016': {
        'max_seq_len': 1024,
        'max_smi_len': 128,
    },
    'pdbbind_v2020': {
        'max_seq_len': 1024,
        'max_smi_len': 128,
    },
    'pdbbind_v2021_time': {
        'max_seq_len': 1024,
        'max_smi_len': 128,
    },
    'pdbbind_v2021_similarity': {
        'max_seq_len': 800,
        'max_smi_len': 256,
    },
    'lp_pdbbind': {
        'max_seq_len': 1024,
        'max_smi_len': 128,
    },
}

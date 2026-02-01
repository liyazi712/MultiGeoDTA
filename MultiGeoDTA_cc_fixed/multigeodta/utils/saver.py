"""
Model and Results Saving Utility for MultiGeoDTA
"""

import yaml
import torch
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Any


class Saver:
    """
    Utility class for saving models, configurations, and results.

    Args:
        output_dir: Directory for saving outputs

    Example:
        >>> saver = Saver('./output/experiment1')
        >>> saver.save_checkpoint(model.state_dict(), 'best_model.pt')
        >>> saver.save_df(results_df, 'predictions.tsv')
        >>> saver.save_config({'lr': 0.001, 'epochs': 100}, 'config.yaml')
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.save_dir = Path(output_dir)

    def mkdir(self):
        """Create output directory if it doesn't exist."""
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def save_checkpoint(self, state_dict: Dict[str, Any],
                       filename: str = 'checkpoint.pt'):
        """
        Save model checkpoint.

        Args:
            state_dict: Model state dictionary
            filename: Checkpoint filename
        """
        self.mkdir()
        filepath = self.save_dir / filename
        torch.save(state_dict, str(filepath))

    def load_checkpoint(self, filename: str = 'checkpoint.pt') -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Model state dictionary
        """
        filepath = self.save_dir / filename
        return torch.load(str(filepath))

    def save_df(self, df: pd.DataFrame, filename: str,
                index: bool = False, float_format: str = '%.6f',
                sep: str = '\t'):
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            index: Whether to save index
            float_format: Float formatting string
            sep: Column separator
        """
        self.mkdir()
        filepath = self.save_dir / filename
        df.to_csv(filepath, float_format=float_format, index=index, sep=sep)

    def save_config(self, config: Dict[str, Any], filename: str,
                   overwrite: bool = True):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            filename: Output filename
            overwrite: Whether to overwrite existing file
        """
        self.mkdir()
        filepath = self.save_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Config file already exists: {filepath}")

        with open(filepath, 'w') as f:
            yaml.dump(config, f, indent=2, default_flow_style=False)

    def load_config(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            filename: Config filename

        Returns:
            Configuration dictionary
        """
        filepath = self.save_dir / filename
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    # Backward compatibility alias
    def save_ckp(self, pt: Dict[str, Any], filename: str = 'checkpoint.pt'):
        """Alias for save_checkpoint (backward compatibility)."""
        self.save_checkpoint(pt, filename)

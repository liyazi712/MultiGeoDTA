"""
Unit Tests for MultiGeoDTA Models

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelImports:
    """Test that all modules can be imported."""

    def test_import_main_package(self):
        import multigeodta
        assert hasattr(multigeodta, '__version__')

    def test_import_models(self):
        from multigeodta.models import DTAModel, ProtGVPModel, DrugGVPModel
        assert DTAModel is not None
        assert ProtGVPModel is not None
        assert DrugGVPModel is not None

    def test_import_data(self):
        from multigeodta.data import DTADataset, DTADataLoader
        assert DTADataset is not None
        assert DTADataLoader is not None

    def test_import_utils(self):
        from multigeodta.utils import Logger, Saver, EarlyStopping, evaluation_metrics
        assert Logger is not None
        assert Saver is not None
        assert EarlyStopping is not None
        assert evaluation_metrics is not None

    def test_import_configs(self):
        from multigeodta.configs import DefaultConfig, get_default_config
        assert DefaultConfig is not None
        assert get_default_config is not None


class TestDTAModel:
    """Test DTAModel functionality."""

    def test_model_creation(self):
        from multigeodta.models import DTAModel
        model = DTAModel()
        assert model is not None

    def test_model_parameters(self):
        from multigeodta.models import DTAModel
        model = DTAModel()
        num_params = model.get_num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_model_custom_config(self):
        from multigeodta.models import DTAModel
        model = DTAModel(
            mlp_dims=[512, 256],
            mlp_dropout=0.3,
        )
        assert model is not None


class TestGVPLayers:
    """Test GVP layer functionality."""

    def test_gvp_layer(self):
        from multigeodta.models.gvp import GVP
        layer = GVP(
            in_dims=(32, 4),
            out_dims=(64, 8),
        )
        assert layer is not None

    def test_layer_norm(self):
        from multigeodta.models.gvp import LayerNorm
        norm = LayerNorm(dims=(32, 4))
        assert norm is not None


class TestEncoders:
    """Test encoder modules."""

    def test_protein_encoder(self):
        from multigeodta.models import ProtGVPModel
        encoder = ProtGVPModel(
            node_in_dim=(6, 3),
            node_h_dim=(64, 32),
            edge_in_dim=(32, 1),
            edge_h_dim=(16, 1),
            num_layers=2,
        )
        assert encoder is not None

    def test_drug_encoder(self):
        from multigeodta.models import DrugGVPModel
        encoder = DrugGVPModel(
            node_in_dim=(86, 1),
            node_h_dim=(64, 32),
            edge_in_dim=(24, 3),
            edge_h_dim=(16, 1),
            num_layers=1,
        )
        assert encoder is not None

    def test_seq_encoder(self):
        from multigeodta.models.encoders import SeqEncoder
        encoder = SeqEncoder(embedding_dim=128)
        assert encoder is not None

        # Test forward pass with dummy input
        batch_size = 4
        seq_len = 100
        seq_input = torch.randint(0, 27, (batch_size, seq_len))
        poc_input = torch.randint(0, 27, (batch_size, seq_len))

        output = encoder(seq_input, poc_input)
        assert output.shape == (batch_size, seq_len, 64)  # embedding_dim // 2

    def test_smile_encoder(self):
        from multigeodta.models.encoders import SmileEncoder
        encoder = SmileEncoder(embedding_dim=128)
        assert encoder is not None

        # Test forward pass
        batch_size = 4
        smi_len = 50
        smi_input = torch.randint(0, 66, (batch_size, smi_len))

        output = encoder(smi_input)
        assert output.shape == (batch_size, smi_len, 64)


class TestMetrics:
    """Test evaluation metrics."""

    def test_evaluation_metrics(self):
        from multigeodta.utils import evaluation_metrics
        import numpy as np

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

        results = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'rmse', 'mae', 'pearson', 'spearman']
        )

        assert 'mse' in results
        assert 'rmse' in results
        assert 'mae' in results
        assert 'pearson' in results
        assert 'spearman' in results

        # Check values are reasonable
        assert results['mse'] >= 0
        assert results['rmse'] >= 0
        assert results['mae'] >= 0
        assert -1 <= results['pearson'] <= 1
        assert -1 <= results['spearman'] <= 1


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_init(self):
        from multigeodta.utils import EarlyStopping
        stopper = EarlyStopping(patience=10)
        assert stopper is not None
        assert stopper.patience == 10
        assert not stopper.early_stop

    def test_early_stopping_update(self):
        from multigeodta.utils import EarlyStopping
        stopper = EarlyStopping(patience=3, higher_better=False)

        # Improving scores
        assert stopper.update(1.0) == True  # First score
        assert stopper.update(0.9) == True  # Better
        assert stopper.update(0.8) == True  # Better

        # Not improving
        assert stopper.update(0.85) == False  # Worse
        assert stopper.update(0.9) == False   # Worse
        assert stopper.update(1.0) == False   # Worse

        # Should trigger early stop
        assert stopper.early_stop == True


class TestConfig:
    """Test configuration functionality."""

    def test_default_config(self):
        from multigeodta.configs import get_default_config

        config = get_default_config('pdbbind_v2016')
        assert config.task == 'pdbbind_v2016'
        assert config.training.n_ensembles == 5
        assert config.training.batch_size == 128

    def test_config_to_dict(self):
        from multigeodta.configs import DefaultConfig

        config = DefaultConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict
        assert 'data' in config_dict


class TestDataset:
    """Test dataset functionality."""

    def test_dataset_creation(self):
        from multigeodta.data import DTADataset

        # Create with empty list
        dataset = DTADataset([])
        assert len(dataset) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

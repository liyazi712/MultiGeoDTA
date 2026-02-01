# MultiGeoDTA

**Multi-modal Geometric Drug-Target Affinity Prediction**

A deep learning framework for predicting drug-target binding affinity by integrating protein structure, sequence, and molecular features using Geometric Vector Perceptrons (GVP) and Mamba State Space Models.

## Overview

MultiGeoDTA combines multiple modalities for accurate binding affinity prediction:

1. **Protein 3D Structure** - Encoded using GVP (Geometric Vector Perceptron) on protein contact graphs
2. **Drug 3D Structure** - Encoded using GVP on molecular graphs
3. **Protein Sequence** - Encoded using Mamba2 State Space Model with local-global fusion
4. **SMILES String** - Encoded using Mamba2 State Space Model

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MultiGeoDTA Model                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Protein    │    │  Drug 3D    │    │  Sequence   │        │
│   │  3D Graph   │    │    Graph    │    │  Encoders   │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │                │
│          ▼                  ▼                  ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Protein    │    │    Drug     │    │  Seq+Pocket │        │
│   │    GVP      │    │    GVP      │    │   Mamba2    │        │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│          │                  │                  │                │
│          └────────────┬─────┴──────────────────┘                │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │  Feature Fusion │                                │
│              │      MLP        │                                │
│              └────────┬────────┘                                │
│                       │                                         │
│                       ▼                                         │
│              ┌─────────────────┐                                │
│              │    Predicted    │                                │
│              │    Affinity     │                                │
│              └─────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation(docs/INSTALL.md is all you need! It's the easiest way to build the virtual environment.)

### Prerequisites

- Python >= 3.8
- CUDA >= 11.8 (for GPU support)
- PyTorch >= 2.0.0

### From Source

```bash
# Clone repository
git clone https://github.com/yourusername/MultiGeoDTA.git
cd MultiGeoDTA

# Create conda environment
conda create -n multigeodta python=3.8
conda activate multigeodta

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch_geometric torch_scatter torch_sparse torch_cluster

# Install DGL
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install Mamba (optional but recommended)
pip install mamba_ssm causal_conv1d

# Install RDKit
pip install rdkit

# Install MultiGeoDTA
pip install -e .
```

### Using pip (coming soon)

```bash
pip install multigeodta
```

## Quick Start

### Training

```python
from multigeodta import DTATrainer
from multigeodta.tasks import PDBBind2016

# Load task and data
task = PDBBind2016(data_dir='./data/pdbbind_v2016')
train_data, valid_data, test_data = task.get_datasets()

# Create trainer
trainer = DTATrainer(
    n_ensembles=5,
    batch_size=128,
    lr=0.0001,
    device=0,
    output_dir='./output/pdbbind_v2016'
)

# Setup data and train
trainer.setup_data(train_data, valid_data, test_data)
trainer.train(n_epochs=100, patience=20)

# Evaluate
results = trainer.evaluate()
print(f"Test RMSE: {results['metrics']['rmse']:.4f}")
print(f"Test Pearson: {results['metrics']['pearson']:.4f}")
```

### Using Command Line

```bash
# Train on PDBBind v2016
python scripts/train.py --task pdbbind_v2016 --device 0

# Train on PDBBind v2021 with similarity-based split
python scripts/train.py \
    --task pdbbind_v2021_similarity \
    --setting new_new \
    --threshold 0.5 \
    --device 0

# Test trained model
python scripts/test.py \
    --task pdbbind_v2016 \
    --checkpoint_dir ./output/pdbbind_v2016

# Virtual screening
python scripts/inference.py \
    --compounds compounds.csv \
    --target_pdb target.pkl.gz \
    --compounds_sdf molecules.pkl.gz \
    --target_sequence "MVLSPADKTN..." \
    --pocket_positions "10,12,15,..." \
    --checkpoint_dir ./output/pdbbind_v2016 \
    --output predictions.csv
```

## Project Structure

```
MultiGeoDTA_cc_fixed/
├── multigeodta/                # Main package
│   ├── __init__.py
│   ├── trainer.py              # Training logic
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── dta_model.py        # Main DTA model
│   │   ├── encoders.py         # Protein/Drug encoders
│   │   └── gvp.py              # GVP layers
│   ├── data/                   # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes
│   │   ├── protein_graph.py    # Protein graph construction
│   │   ├── mol_graph.py        # Molecule graph construction
│   │   └── constants.py        # Constants and vocabularies
│   ├── configs/                # Configuration management
│   │   ├── __init__.py
│   │   └── default_config.py
│   ├── tasks/                  # Task definitions
│   │   ├── __init__.py
│   │   ├── pdbbind.py          # PDBBind tasks
│   │   └── virtual_screening.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logger.py
│       ├── saver.py
│       ├── early_stopping.py
│       └── metrics.py
├── scripts/                    # Entry point scripts
│   ├── train.py
│   ├── test.py
│   └── inference.py
├── examples/                   # Example notebooks and scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
```

## Supported Tasks

| Task | Description | Split |
|------|-------------|-------|
| `pdbbind_v2016` | PDBBind v2016 benchmark | Random |
| `pdbbind_v2020` | PDBBind v2020 benchmark | Random |
| `pdbbind_v2021_time` | PDBBind v2021 | Time-based |
| `pdbbind_v2021_similarity` | PDBBind v2021 | Similarity-based |
| `lp_pdbbind` | LP-PDBBind benchmark | Random |

### Similarity-based Split Settings

For `pdbbind_v2021_similarity`, you can choose:
- **Setting**: `new_compound`, `new_protein`, `new_new`
- **Threshold**: `0.3`, `0.4`, `0.5`, `0.6`

## Data Availability

Pre-processed datasets can be downloaded from Hugging Face:
- [MultiGeoDTA Datasets](https://huggingface.co/datasets/your-dataset)

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Pearson**: Pearson Correlation Coefficient
- **Spearman**: Spearman Rank Correlation
- **CI**: Concordance Index
- **rm²**: Modified R-squared

## Citation

If you use MultiGeoDTA in your research, please cite:

```bibtex
@article{multigeodta2024,
  title={MultiGeoDTA: Multi-modal Geometric Drug-Target Affinity Prediction},
  author={Li, Yazi and others},
  journal={...},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Yazi Li
- **Email**: liyazi126@126.com

## Acknowledgments

- GVP implementation adapted from [drorlab/gvp-pytorch](https://github.com/drorlab/gvp-pytorch)
- Mamba implementation from [state-spaces/mamba](https://github.com/state-spaces/mamba)

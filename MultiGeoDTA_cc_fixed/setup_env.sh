#!/bin/bash
# MultiGeoDTA Environment Setup Script
# This script helps you set up the environment for MultiGeoDTA

set -e

echo "========================================"
echo "MultiGeoDTA Environment Setup"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Configuration
ENV_NAME="${1:-multigeodta}"
PYTHON_VERSION="3.8"
CUDA_VERSION="${2:-118}"  # Default CUDA 11.8

echo "Creating conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo "CUDA version: $CUDA_VERSION"
echo ""

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
fi

echo "Installing PyTorch Geometric..."
pip install torch_geometric

# Install PyG dependencies based on PyTorch and CUDA versions
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
else
    pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html
fi

echo "Installing DGL..."
if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install dgl -f https://data.dgl.ai/wheels/repo.html
else
    pip install dgl -f https://data.dgl.ai/wheels/cu${CUDA_VERSION}/repo.html
fi

echo "Installing RDKit..."
pip install rdkit

echo "Installing Mamba (optional, will fallback to LSTM if fails)..."
pip install mamba_ssm causal_conv1d || echo "Warning: Mamba installation failed. LSTM fallback will be used."

echo "Installing other dependencies..."
pip install numpy scipy pandas scikit-learn tqdm pyyaml joblib tensorboard

echo "Installing MultiGeoDTA..."
pip install -e .

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation:"
echo "  python -c \"from multigeodta import DTAModel; print('Success!')\""
echo ""
echo "To run training:"
echo "  python scripts/train.py --task pdbbind_v2016 --device 0"

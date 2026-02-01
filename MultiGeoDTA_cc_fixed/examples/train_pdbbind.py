#!/usr/bin/env python
"""
Example: Training MultiGeoDTA on PDBBind v2016

This example demonstrates how to:
1. Load and prepare PDBBind data
2. Configure the model and trainer
3. Train the model with early stopping
4. Evaluate on test set
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random

from multigeodta import DTATrainer, DTAModel
from multigeodta.tasks import PDBBind2016
from multigeodta.configs import get_default_config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Set random seed
    set_seed(42)

    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load configuration
    config = get_default_config('pdbbind_v2016')
    print(f"Configuration loaded for task: {config.task}")

    # Load data
    print("\nLoading PDBBind v2016 data...")
    task = PDBBind2016(data_dir='./data/pdbbind_v2016')
    train_dataset, valid_dataset, test_dataset = task.get_datasets()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DTATrainer(
        model_config={
            'mlp_dims': config.model.mlp_dims,
            'mlp_dropout': config.model.mlp_dropout,
        },
        n_ensembles=config.training.n_ensembles,
        batch_size=config.training.batch_size,
        lr=config.training.lr,
        device=device,
        output_dir='./output/pdbbind_v2016_example',
        save_log=True,
        save_checkpoint=True,
    )

    # Setup data
    trainer.setup_data(train_dataset, valid_dataset, test_dataset)

    # Print model info
    print(f"\nModel architecture:")
    print(f"  Parameters: {trainer.models[0].get_num_parameters():,}")

    # Train
    print("\nStarting training...")
    trainer.train(
        n_epochs=config.training.n_epochs,
        patience=config.training.patience,
        eval_freq=config.training.eval_freq,
        monitoring_score='mse',
    )

    # Evaluate
    print("\nFinal evaluation on test set...")
    results = trainer.evaluate(save_prediction=True)

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nTest Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nOutput saved to: ./output/pdbbind_v2016_example")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Training Script for MultiGeoDTA

Usage:
    python scripts/train.py --task pdbbind_v2016 --device 0
    python scripts/train.py --task pdbbind_v2021_similarity --setting new_new --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multigeodta import DTATrainer
from multigeodta.configs import get_default_config
from multigeodta.tasks import (
    PDBBind2016, PDBBind2020, PDBBind2021Time,
    PDBBind2021Similarity, LPPDBBind
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiGeoDTA model')

    # Task configuration
    parser.add_argument('--task', type=str, default='pdbbind_v2016',
                       choices=['pdbbind_v2016', 'pdbbind_v2020', 'pdbbind_v2021_time',
                               'pdbbind_v2021_similarity', 'lp_pdbbind'],
                       help='Task/dataset name')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: ./data/{task})')
    parser.add_argument('--setting', type=str, default='new_new',
                       choices=['new_compound', 'new_protein', 'new_new'],
                       help='Split setting for similarity-based tasks')
    parser.add_argument('--threshold', type=float, default=0.5,
                       choices=[0.3, 0.4, 0.5, 0.6],
                       help='Similarity threshold')

    # Training configuration
    parser.add_argument('--n_ensembles', type=int, default=5,
                       help='Number of ensemble models')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--eval_freq', type=int, default=1,
                       help='Evaluation frequency (epochs)')

    # Model configuration
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512],
                       help='MLP layer dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
                       help='MLP dropout rate')

    # Device configuration
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device ID')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel training')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--save_log', action='store_true', default=True,
                       help='Save training log')
    parser.add_argument('--save_checkpoint', action='store_true', default=True,
                       help='Save model checkpoints')
    parser.add_argument('--save_prediction', action='store_true', default=True,
                       help='Save predictions')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def get_task(args):
    """Get task instance based on arguments."""
    data_dir = args.data_dir or f'./data/{args.task}'

    task_classes = {
        'pdbbind_v2016': PDBBind2016,
        'pdbbind_v2020': PDBBind2020,
        'pdbbind_v2021_time': PDBBind2021Time,
        'pdbbind_v2021_similarity': PDBBind2021Similarity,
        'lp_pdbbind': LPPDBBind,
    }

    if args.task == 'pdbbind_v2021_similarity':
        return task_classes[args.task](
            data_dir=data_dir,
            setting=args.setting,
            threshold=args.threshold
        )
    else:
        return task_classes[args.task](data_dir=data_dir)


def main():
    args = parse_args()

    # Set random seed
    import torch
    import numpy as np
    import random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup output directory
    output_dir = Path(args.output_dir) / args.task
    if args.task == 'pdbbind_v2021_similarity':
        output_dir = output_dir / args.setting / str(args.threshold)

    # Load task and data
    print(f"Loading task: {args.task}")
    task = get_task(args)
    train_dataset, valid_dataset, test_dataset = task.get_datasets()

    # Create trainer
    model_config = {
        'mlp_dims': args.mlp_dims,
        'mlp_dropout': args.mlp_dropout,
    }

    trainer = DTATrainer(
        model_config=model_config,
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=str(output_dir),
        save_log=args.save_log,
        save_checkpoint=args.save_checkpoint,
        parallel=args.parallel,
    )

    # Setup data
    trainer.setup_data(train_dataset, valid_dataset, test_dataset)

    # Save configuration
    trainer.save_config(vars(args), 'config.yaml')

    # Train
    print(f"\nStarting training...")
    trainer.train(
        n_epochs=args.n_epochs,
        patience=args.patience,
        eval_freq=args.eval_freq,
        monitoring_score='mse',
        test_after_train=True,
    )

    # Final evaluation
    print(f"\nFinal evaluation...")
    results = trainer.evaluate(
        save_prediction=args.save_prediction,
        save_name='predictions.tsv'
    )

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

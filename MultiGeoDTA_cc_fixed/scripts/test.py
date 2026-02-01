#!/usr/bin/env python
"""
Testing Script for MultiGeoDTA

Evaluate trained models on test sets.

Usage:
    python scripts/test.py --task pdbbind_v2016 --checkpoint_dir ./output/pdbbind_v2016
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multigeodta import DTATrainer
from multigeodta.tasks import (
    PDBBind2016, PDBBind2020, PDBBind2021Time,
    PDBBind2021Similarity, LPPDBBind
)


def parse_args():
    parser = argparse.ArgumentParser(description='Test MultiGeoDTA model')

    parser.add_argument('--task', type=str, required=True,
                       choices=['pdbbind_v2016', 'pdbbind_v2020', 'pdbbind_v2021_time',
                               'pdbbind_v2021_similarity', 'lp_pdbbind'],
                       help='Task/dataset name')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory')
    parser.add_argument('--setting', type=str, default='new_new',
                       help='Split setting for similarity tasks')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold')
    parser.add_argument('--n_ensembles', type=int, default=5,
                       help='Number of ensemble models')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results')

    return parser.parse_args()


def get_task(args):
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

    print(f"Loading task: {args.task}")
    task = get_task(args)
    train_dataset, valid_dataset, test_dataset = task.get_datasets()

    # Create trainer
    trainer = DTATrainer(
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        save_log=True,
        save_checkpoint=False,
    )

    # Setup test data
    trainer.setup_data(train_dataset, valid_dataset, test_dataset)

    # Load checkpoints
    print(f"Loading checkpoints from: {args.checkpoint_dir}")
    trainer.load_checkpoints(args.checkpoint_dir)

    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate(save_prediction=True, save_name='test_predictions.tsv')

    print(f"\nTest Results:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()

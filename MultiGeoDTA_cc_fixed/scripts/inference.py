#!/usr/bin/env python
"""
Inference Script for MultiGeoDTA

Perform virtual screening or predict binding affinities for new compounds.

Usage:
    python scripts/inference.py \
        --compounds zinc_compounds.csv \
        --target_pdb target_pocket.pkl.gz \
        --compounds_sdf zinc_molecules.pkl.gz \
        --checkpoint_dir ./output/pdbbind_v2016 \
        --output results.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from multigeodta import DTATrainer
from multigeodta.tasks import VirtualScreeningTask
from multigeodta.data import DTADataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='MultiGeoDTA Inference')

    # Input data
    parser.add_argument('--compounds', type=str, required=True,
                       help='CSV file with compound information')
    parser.add_argument('--target_pdb', type=str, required=True,
                       help='Target protein structure file')
    parser.add_argument('--compounds_sdf', type=str, required=True,
                       help='Compound structures file')
    parser.add_argument('--target_sequence', type=str, required=True,
                       help='Target protein sequence')
    parser.add_argument('--pocket_positions', type=str, required=True,
                       help='Comma-separated pocket residue positions')

    # Model
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--n_ensembles', type=int, default=5,
                       help='Number of ensemble models')

    # Processing
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--device', type=int, default=0,
                       help='CUDA device')

    # Output
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file for predictions')
    parser.add_argument('--top_k', type=int, default=None,
                       help='Only output top K compounds')

    return parser.parse_args()


def main():
    args = parse_args()

    # Parse pocket positions
    pocket_positions = [int(x) for x in args.pocket_positions.split(',')]

    # Create task
    print("Loading data...")
    task = VirtualScreeningTask(
        compounds_csv=args.compounds,
        target_pdb=args.target_pdb,
        compounds_sdf=args.compounds_sdf,
        target_sequence=args.target_sequence,
        pocket_positions=pocket_positions,
    )

    # Get dataset
    dataset = task.get_dataset()
    data_loader = DTADataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create trainer and load model
    print("Loading model...")
    trainer = DTATrainer(
        n_ensembles=args.n_ensembles,
        batch_size=args.batch_size,
        device=args.device,
        output_dir='./inference_output',
        save_log=False,
        save_checkpoint=False,
    )

    trainer.load_checkpoints(args.checkpoint_dir)

    # Run inference
    print("Running inference...")
    predictions = trainer.predict(data_loader)

    # Create results
    results = task.create_results_dataframe(predictions)

    # Filter top K if specified
    if args.top_k:
        results = results.head(args.top_k)

    # Save results
    results.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

    # Print summary
    print(f"\nScreened {len(predictions)} compounds")
    print(f"Top 10 compounds:")
    print(results.head(10).to_string(index=False))


if __name__ == '__main__':
    main()

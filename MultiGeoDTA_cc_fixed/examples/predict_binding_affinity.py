#!/usr/bin/env python
"""
Example: Predicting Binding Affinity for New Compounds

This example demonstrates how to:
1. Load a pre-trained model
2. Prepare input data (protein + compounds)
3. Make predictions
4. Analyze results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from multigeodta import DTATrainer
from multigeodta.tasks import VirtualScreeningTask


def main():
    # Example target protein sequence (CB1R)
    target_sequence = """
    GENFMDIECFMVLNPSQQLAIAVLSLTLGTFTVLENLLVLCVILHSRSLRCRPSYHFIGSLAVADLLGSVIFVYSFIDFHVFHRKDSRNVFLFKLGGVTASFTASVGSLFLAAIDRYISIHRPLAYKRIVTRPKAVVAFCLMWTIAIVIAVLPLLGWNCEKLQSVCSDIFPHIDKTYLMFWIGVVSVLLLFIVYAYMYILWKAHSHAVAKALIVYGSTTGNTEYTAETIARELADAGYEVDSRDAASVEAGGLFEGFDLVLLGCSTWGDDSIELQDDFIPLFDSLEETGAQGRKVACFGCGDSSWEYFCGAVDAIEEKLKNLGAEIVQDGLRIDGDPRAARDDIVGWAHDVRGAIPDQARMDIELAKTLVLILVVLIICWGPLLAIMVYDVFGKMNKLIKTVFAFCSMLCLLNSTVNPIIYALRSKDLRHAFRSMFPS
    """.replace('\n', '').replace(' ', '')

    # Pocket positions
    pocket_positions = [
        10, 12, 72, 75, 76, 79, 80, 86, 91, 94, 95, 98, 99, 102,
        168, 169, 170, 171, 173, 177, 178, 181, 380, 383, 387,
        400, 401, 403, 404, 407, 410
    ]

    print("="*60)
    print("MultiGeoDTA Binding Affinity Prediction Example")
    print("="*60)

    print(f"\nTarget: CB1R (Cannabinoid receptor 1)")
    print(f"Sequence length: {len(target_sequence)}")
    print(f"Pocket residues: {len(pocket_positions)}")

    # Note: In real usage, you would load actual data files
    # Here we demonstrate the API

    print("\n" + "-"*60)
    print("To use this example with real data:")
    print("-"*60)
    print("""
    1. Prepare your data files:
       - compounds.csv: CSV with 'compound_id' and 'smiles' columns
       - target.pkl.gz: Pickled protein structure
       - molecules.pkl.gz: Pickled molecule structures

    2. Initialize the task:
       task = VirtualScreeningTask(
           compounds_csv='compounds.csv',
           target_pdb='target.pkl.gz',
           compounds_sdf='molecules.pkl.gz',
           target_sequence=target_sequence,
           pocket_positions=pocket_positions,
       )

    3. Load trained model and predict:
       trainer = DTATrainer(
           n_ensembles=5,
           device=0,
       )
       trainer.load_checkpoints('./output/pdbbind_v2016')

       dataset = task.get_dataset()
       predictions = trainer.predict(DataLoader(dataset))

    4. Analyze results:
       results = task.create_results_dataframe(predictions)
       print(results.head(10))  # Top 10 compounds
    """)

    print("\nExample output format:")
    print("-"*60)

    # Create mock results for demonstration
    mock_results = pd.DataFrame({
        'compound_id': [f'ZINC{i:08d}' for i in range(10)],
        'smiles': [
            'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
            'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
            'COC1=CC=CC=C1OCC(CNC(C)C)O',
            'CC(C)NCC(COC1=CC=CC2=CC=CC=C21)O',
            'C1CN(CCN1)C2=NC3=CC=CC=C3N=C2N',
            'CN1C2=C(C=C(C=C2)Cl)C(=N1)C3=CC=NC=C3',
            'CC1=CC(=NO1)C2=CC=CC=C2',
            'CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C',
            'C1=CC=C(C=C1)CCNC(=O)C2=CC=C(C=C2)O',
            'CC(C)(C)C1=CC=C(C=C1)C(=O)NC2=CC=CC=C2'
        ],
        'predicted_affinity': [8.5, 8.2, 7.9, 7.8, 7.5, 7.3, 7.1, 6.9, 6.7, 6.5],
        'rank': range(1, 11)
    })

    print(mock_results.to_string(index=False))

    print("\n" + "="*60)
    print("For complete documentation, visit:")
    print("https://github.com/yourusername/MultiGeoDTA")
    print("="*60)


if __name__ == '__main__':
    main()

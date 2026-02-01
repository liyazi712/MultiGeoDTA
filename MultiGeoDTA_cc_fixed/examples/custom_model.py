#!/usr/bin/env python
"""
Example: Using Custom Model Configuration

This example demonstrates how to:
1. Create a model with custom architecture
2. Inspect model components
3. Use individual encoders
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from multigeodta.models import DTAModel, ProtGVPModel, DrugGVPModel, SeqEncoder, SmileEncoder
from multigeodta.configs import ModelConfig


def main():
    print("="*60)
    print("MultiGeoDTA Custom Model Configuration Example")
    print("="*60)

    # Example 1: Default model
    print("\n1. Creating default model...")
    model = DTAModel()
    print(f"   Total parameters: {model.get_num_parameters():,}")

    # Example 2: Custom model configuration
    print("\n2. Creating custom model...")
    custom_model = DTAModel(
        # Drug encoder
        drug_node_in_dim=(86, 1),
        drug_node_h_dims=(256, 128),  # Larger hidden dims
        drug_edge_in_dim=(24, 3),
        drug_edge_h_dims=(64, 2),

        # Protein encoder
        prot_node_in_dim=(6, 3),
        prot_node_h_dims=(256, 128),  # Larger hidden dims
        prot_edge_in_dim=(32, 1),
        prot_edge_h_dims=(64, 2),

        # MLP
        mlp_dims=[2048, 1024, 512],  # Deeper MLP
        mlp_dropout=0.3,

        # Sequence encoder
        seq_embedding_dim=512,  # Larger embedding
    )
    print(f"   Total parameters: {custom_model.get_num_parameters():,}")

    # Example 3: Using configuration dataclass
    print("\n3. Using ModelConfig dataclass...")
    config = ModelConfig(
        drug_node_h_dims=(128, 64),
        prot_node_h_dims=(128, 64),
        mlp_dims=[1024, 512],
        mlp_dropout=0.25,
    )
    print(f"   Config: {config}")

    # Example 4: Inspect model components
    print("\n4. Model components:")
    print(f"   - Drug GVP Encoder: {type(model.drug_GVP).__name__}")
    print(f"   - Protein GVP Encoder: {type(model.prot_GVP).__name__}")
    print(f"   - Sequence Encoder: {type(model.seq_encoder).__name__}")
    print(f"   - SMILES Encoder: {type(model.smile_encoder).__name__}")

    # Example 5: Using individual encoders
    print("\n5. Using individual encoders...")

    # Protein GVP encoder
    prot_encoder = ProtGVPModel(
        node_in_dim=(6, 3),
        node_h_dim=(128, 64),
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_layers=3,
    )
    print(f"   Protein encoder params: {sum(p.numel() for p in prot_encoder.parameters()):,}")

    # Drug GVP encoder
    drug_encoder = DrugGVPModel(
        node_in_dim=(86, 1),
        node_h_dim=(128, 64),
        edge_in_dim=(24, 3),
        edge_h_dim=(32, 1),
        num_layers=1,
    )
    print(f"   Drug encoder params: {sum(p.numel() for p in drug_encoder.parameters()):,}")

    # Sequence encoder
    seq_encoder = SeqEncoder(embedding_dim=256)
    print(f"   Sequence encoder params: {sum(p.numel() for p in seq_encoder.parameters()):,}")

    # Example 6: Freeze/unfreeze encoders
    print("\n6. Freezing/unfreezing encoders...")
    model.freeze_encoder('drug')
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   After freezing drug encoder: {trainable_after_freeze:,} trainable params")

    model.unfreeze_encoder('drug')
    trainable_after_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   After unfreezing: {trainable_after_unfreeze:,} trainable params")

    print("\n" + "="*60)
    print("Example complete!")
    print("="*60)


if __name__ == '__main__':
    main()

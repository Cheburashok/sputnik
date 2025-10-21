"""
Example training script for land segmentation model.
This demonstrates how to use the modular training pipeline.
"""
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.land_segmentation import (
    ModelConfig,
    DataConfig,
    TrainingConfig
)
from analysis.land_segmentation.train import train


def main():
    """Main training function."""

    # Configure model
    model_config = ModelConfig(
        encoder='resnet50',
        encoder_weights='imagenet',
        activation='sigmoid'
    )

    # Configure data
    data_config = DataConfig(
        data_dir='/path/to/deepglobe/dataset',
        metadata_csv='/path/to/metadata.csv',
        train_img_size=1024,
        val_img_size=1024,
        train_val_split=0.1,
        random_seed=42
    )

    # Configure training
    training_config = TrainingConfig(
        batch_size=12,
        num_epochs=40,
        learning_rate=0.00008,
        num_workers=4,
        device='cuda',
        checkpoint_dir='checkpoints',
        save_every_n_epochs=1,
        loss_type='dice'
    )

    # Start training
    print("Starting training...")
    print(f"Model: {model_config.encoder}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.num_epochs}")
    print(f"Device: {training_config.device}")

    train(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {training_config.checkpoint_dir}")


if __name__ == '__main__':
    main()

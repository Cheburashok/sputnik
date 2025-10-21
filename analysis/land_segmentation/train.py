"""
Training script for land segmentation model.
Provides clean training loops and model setup.
"""
import os
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from .config import ModelConfig, DataConfig, TrainingConfig
from .datasets import LandCoverDataset
from .transforms import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing
)
from .utils import save_checkpoint


def create_model(config: ModelConfig, num_classes: int):
    """
    Create segmentation model.

    Args:
        config: Model configuration
        num_classes: Number of output classes

    Returns:
        PyTorch model
    """
    model = smp.DeepLabV3Plus(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        classes=num_classes,
        activation=config.activation,
    )
    return model


def prepare_dataloaders(
    data_config: DataConfig,
    training_config: TrainingConfig,
    preprocessing_fn=None
):
    """
    Prepare train and validation dataloaders.

    Args:
        data_config: Data configuration
        training_config: Training configuration
        preprocessing_fn: Preprocessing function from encoder

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load metadata
    metadata_df = pd.read_csv(data_config.metadata_csv)
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]

    # Update paths to be absolute
    data_dir = data_config.data_dir
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(
        lambda p: os.path.join(data_dir, p)
    )
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(
        lambda p: os.path.join(data_dir, p)
    )

    # Shuffle and split
    metadata_df = metadata_df.sample(frac=1, random_state=data_config.random_seed).reset_index(drop=True)
    val_df = metadata_df.sample(frac=data_config.train_val_split, random_state=data_config.random_seed)
    train_df = metadata_df.drop(val_df.index)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    # Create datasets
    train_dataset = LandCoverDataset(
        train_df,
        augmentation=get_training_augmentation(data_config.train_img_size),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=data_config.class_rgb_values,
    )

    val_dataset = LandCoverDataset(
        val_df,
        augmentation=get_validation_augmentation(data_config.val_img_size),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=data_config.class_rgb_values,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str,
) -> dict:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        pbar.set_postfix({'loss': total_loss / n_samples})

    return {'loss': total_loss / n_samples}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: str,
) -> dict:
    """
    Validate for one epoch.

    Args:
        model: PyTorch model
        loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0

    pbar = tqdm(loader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = loss_fn(logits, masks)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        pbar.set_postfix({'loss': total_loss / n_samples})

    return {'loss': total_loss / n_samples}


def train(
    model_config: ModelConfig,
    data_config: DataConfig,
    training_config: TrainingConfig,
    resume_from: str = None,
):
    """
    Main training function.

    Args:
        model_config: Model configuration
        data_config: Data configuration
        training_config: Training configuration
        resume_from: Optional checkpoint path to resume from
    """
    # Setup
    device = training_config.device
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    # Create model
    num_classes = len(data_config.class_names)
    model = create_model(model_config, num_classes)
    model = model.to(device)

    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        model_config.encoder,
        model_config.encoder_weights
    )

    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        data_config,
        training_config,
        preprocessing_fn
    )

    # Setup loss and optimizer
    if training_config.loss_type == 'dice':
        loss_fn = smp.utils.losses.DiceLoss()
    else:
        raise ValueError(f"Unknown loss type: {training_config.loss_type}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        train_logs = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_logs = validate_one_epoch(model, val_loader, loss_fn, device)

        print(f"Train loss: {train_logs['loss']:.4f} | Val loss: {val_logs['loss']:.4f}")

        # Save checkpoint
        if (epoch + 1) % training_config.save_every_n_epochs == 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f'model_epoch_{epoch + 1}.pth'
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_logs['loss'],
                val_loss=val_logs['loss'],
                filepath=checkpoint_path
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_logs['loss'] < best_val_loss:
            best_val_loss = val_logs['loss']
            best_path = os.path.join(training_config.checkpoint_dir, 'best_model.pth')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=train_logs['loss'],
                val_loss=val_logs['loss'],
                filepath=best_path
            )
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")

    print("\nTraining complete!")


if __name__ == '__main__':
    # Example usage
    model_config = ModelConfig()
    data_config = DataConfig(
        data_dir='/path/to/dataset',
        metadata_csv='/path/to/metadata.csv'
    )
    training_config = TrainingConfig()

    train(model_config, data_config, training_config)

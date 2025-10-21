"""
Configuration file for land segmentation model.
Centralized configuration for all hyperparameters and settings.
"""
from dataclasses import dataclass
from typing import List, Tuple
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    encoder: str = 'resnet50'
    encoder_weights: str = 'imagenet'
    activation: str = 'sigmoid'  # None for logits, 'softmax2d' for multiclass


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    # Class information
    class_names: List[str] = None
    class_rgb_values: List[List[int]] = None

    # Image sizes
    train_img_size: int = 1024
    val_img_size: int = 1024

    # Data paths
    data_dir: str = None
    metadata_csv: str = None

    # Data split
    train_val_split: float = 0.1  # 10% for validation
    random_seed: int = 42

    def __post_init__(self):
        # Default class configuration for DeepGlobe dataset
        if self.class_names is None:
            self.class_names = [
                'urban_land',
                'agriculture_land',
                'rangeland',
                'forest_land',
                'water',
                'barren_land',
                'unknown'
            ]

        if self.class_rgb_values is None:
            self.class_rgb_values = [
                [0, 255, 255],    # urban_land (cyan)
                [255, 255, 0],    # agriculture_land (yellow)
                [255, 0, 255],    # rangeland (magenta)
                [0, 255, 0],      # forest_land (green)
                [0, 0, 255],      # water (blue)
                [255, 255, 255],  # barren_land (white)
                [0, 0, 0]         # unknown (black)
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 12
    num_epochs: int = 40
    learning_rate: float = 0.00008
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 1

    # Loss and metrics
    loss_type: str = 'dice'  # 'dice', 'ce', 'focal'


@dataclass
class InferenceConfig:
    """Inference configuration."""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    pad_multiple: int = 32  # Pad images to multiple of this value
    pad_mode: str = 'reflect'  # 'reflect', 'constant', 'edge'

    # Visualization
    show_soft_predictions: bool = False
    save_visualizations: bool = True

    # Other class detection threshold
    other_class_ratio_threshold: float = 0.3

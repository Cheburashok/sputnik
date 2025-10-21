"""
Land Segmentation Package

A modular, well-structured package for training and inference of land cover
segmentation models using satellite imagery.

This package provides:
- Clean PyTorch Dataset implementations
- Training and evaluation pipelines
- Pythonic inference API (replaces unpythonic test_folder function)
- Comprehensive configuration management
- Time-series analysis for land cover changes
"""

__version__ = '1.0.0'

from .config import ModelConfig, DataConfig, TrainingConfig, InferenceConfig
from .datasets import LandCoverDataset, InferenceDataset, ImageFolderDataset
from .predict import Predictor, create_predictor
from .utils import (
    one_hot_encode,
    reverse_one_hot,
    colour_code_segmentation,
    visualize,
    pad_to_multiple,
    crop_from_pad,
    save_checkpoint,
    load_checkpoint,
)
from .transforms import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)

__all__ = [
    # Configuration
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'InferenceConfig',

    # Datasets
    'LandCoverDataset',
    'InferenceDataset',
    'ImageFolderDataset',

    # Prediction
    'Predictor',
    'create_predictor',

    # Utils
    'one_hot_encode',
    'reverse_one_hot',
    'colour_code_segmentation',
    'visualize',
    'pad_to_multiple',
    'crop_from_pad',
    'save_checkpoint',
    'load_checkpoint',

    # Transforms
    'get_training_augmentation',
    'get_validation_augmentation',
    'get_preprocessing',
]

"""
Data augmentation and preprocessing transforms for land segmentation.
Uses albumentations library for efficient augmentations.
"""
import albumentations as album
from typing import Optional, Callable


def get_training_augmentation(img_size: int = 1024) -> album.Compose:
    """
    Get training augmentation pipeline for satellite images.

    Args:
        img_size: Size to crop/resize images to

    Returns:
        Albumentations composition of transforms
    """
    return album.Compose([
        # Geometric augmentations
        album.RandomCrop(height=img_size, width=img_size, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.RandomRotate90(p=0.2),

        # Color and brightness adjustments
        album.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.2
        ),
        album.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.2
        ),
        album.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.2
        ),

        # Blur and noise
        album.OneOf([
            album.GaussianBlur(blur_limit=(3, 5), p=0.2),
            album.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.3),

        # Other augmentations
        album.RandomGamma(gamma_limit=(80, 120), p=0.3),
        album.CLAHE(clip_limit=2.0, p=0.2),
    ])


def get_validation_augmentation(img_size: int = 1024) -> album.Compose:
    """
    Get validation augmentation pipeline (minimal transforms).

    Args:
        img_size: Size to crop images to

    Returns:
        Albumentations composition of transforms
    """
    return album.Compose([
        album.CenterCrop(height=img_size, width=img_size, always_apply=True),
    ])


def get_inference_augmentation() -> album.Compose:
    """
    Get inference augmentation pipeline (no transforms, just passthrough).

    Returns:
        Empty albumentations composition
    """
    return album.Compose([])


def to_tensor(x, **kwargs):
    """
    Convert numpy array from HWC to CHW format (PyTorch convention).

    Args:
        x: Input array (H x W x C)

    Returns:
        Array in CHW format
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn: Optional[Callable] = None) -> album.Compose:
    """
    Construct preprocessing transform pipeline.

    Args:
        preprocessing_fn: Optional normalization function
                         (e.g., from segmentation_models_pytorch encoder)

    Returns:
        Albumentations composition of preprocessing transforms
    """
    transforms = []

    if preprocessing_fn:
        transforms.append(album.Lambda(image=preprocessing_fn))

    transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(transforms)

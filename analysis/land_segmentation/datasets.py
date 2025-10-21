"""
PyTorch Dataset classes for land segmentation.
Provides clean, modular dataset implementations following PyTorch best practices.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset

from .utils import one_hot_encode, pad_to_multiple


class LandCoverDataset(Dataset):
    """
    DeepGlobe Land Cover Classification Dataset.

    Reads satellite images and segmentation masks, applies augmentation and preprocessing.

    Args:
        df: DataFrame containing 'sat_image_path' and 'mask_path' columns
        class_rgb_values: List of RGB values for each class
        augmentation: Albumentations composition for data augmentation
        preprocessing: Albumentations composition for preprocessing (normalization, etc.)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        class_rgb_values: Optional[List[List[int]]] = None,
        augmentation: Optional[Callable] = None,
        preprocessing: Optional[Callable] = None,
    ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # Read image and mask
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)

        # One-hot encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self) -> int:
        return len(self.image_paths)


class InferenceDataset(Dataset):
    """
    Dataset for inference on a folder of images.

    This is the Pythonic replacement for the test_folder function.
    Properly extends torch.utils.data.Dataset for clean inference patterns.

    Args:
        image_folder: Path to folder containing images
        preprocessing_fn: Preprocessing function (e.g., from SMP encoder)
        file_extensions: Tuple of file extensions to look for (default: png only)
        pad_multiple: Pad images to be divisible by this value (default: 32)
        pad_mode: Padding mode ('reflect', 'constant', 'edge')
    """

    def __init__(
        self,
        image_folder: Union[str, Path],
        preprocessing_fn: Optional[Callable] = None,
        file_extensions: Tuple[str, ...] = ('*.png',),
        pad_multiple: int = 32,
        pad_mode: str = 'reflect',
    ):
        self.image_folder = Path(image_folder)
        self.preprocessing_fn = preprocessing_fn
        self.pad_multiple = pad_multiple
        self.pad_mode = pad_mode

        # Collect all image paths
        self.image_paths = []
        for ext in file_extensions:
            self.image_paths.extend(sorted(self.image_folder.glob(ext)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder} with extensions {file_extensions}")

        # Store padding info for each image (computed during __getitem__)
        self.pad_info_cache = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            Tuple of (preprocessed_tensor, metadata) where metadata contains:
                - 'path': Path to the image
                - 'original_shape': Original image shape (H, W, C)
                - 'pad_info': Padding information (top, left, H, W)
                - 'original_image': Original RGB image as uint8 numpy array
        """
        img_path = self.image_paths[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32)
        original_rgb = img_np.astype(np.uint8)
        original_shape = img_np.shape

        # Pad to multiple of pad_multiple
        padded_np, pad_info = pad_to_multiple(
            img_np,
            k=self.pad_multiple,
            mode=self.pad_mode
        )

        # Normalize to [0, 1]
        inp = padded_np / 255.0

        # Apply preprocessing function if provided
        if self.preprocessing_fn:
            inp = self.preprocessing_fn(inp)

        # Convert to CHW format and create tensor
        inp = np.transpose(inp, (2, 0, 1))
        tensor = torch.from_numpy(inp).float()

        # Store metadata
        metadata = {
            'path': img_path,
            'original_shape': original_shape,
            'pad_info': pad_info,
            'original_image': original_rgb,
        }

        return tensor, metadata

    def get_filename(self, idx: int) -> str:
        """Get filename for a given index."""
        return self.image_paths[idx].name


class ImageFolderDataset(Dataset):
    """
    Simple dataset for loading images from a folder (no masks).

    Useful for visualization or when you only need to load images without labels.

    Args:
        image_folder: Path to folder containing images
        file_extensions: Tuple of file extensions to look for
        transform: Optional transform to apply
    """

    def __init__(
        self,
        image_folder: Union[str, Path],
        file_extensions: Tuple[str, ...] = ('*.png', '*.jpg', '*.jpeg'),
        transform: Optional[Callable] = None,
    ):
        self.image_folder = Path(image_folder)
        self.transform = transform

        # Collect all image paths
        self.image_paths = []
        for ext in file_extensions:
            self.image_paths.extend(sorted(self.image_folder.glob(ext)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_folder}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Path]:
        img_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, img_path

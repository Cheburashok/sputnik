"""
Utility functions for land segmentation.
Contains helper functions for encoding, visualization, padding, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch


def one_hot_encode(label: np.ndarray, label_values: List[List[int]]) -> np.ndarray:
    """
    Convert a segmentation image label array to one-hot format.

    Args:
        label: The 2D/3D array segmentation image label (H x W x C)
        label_values: List of RGB values for each class

    Returns:
        One-hot encoded array (H x W x num_classes)
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image: np.ndarray) -> np.ndarray:
    """
    Transform a 2D array in one-hot format to a 2D array with class indices.

    Args:
        image: One-hot format image (H x W x num_classes) or (num_classes x H x W)

    Returns:
        2D array with class indices (H x W)
    """
    # Handle both HWC and CHW formats
    if len(image.shape) == 3:
        if image.shape[-1] < image.shape[0]:  # CHW format
            image = np.transpose(image, (1, 2, 0))
    return np.argmax(image, axis=-1)


def colour_code_segmentation(image: np.ndarray, label_values: List[List[int]]) -> np.ndarray:
    """
    Apply color coding to a segmentation mask.

    Args:
        image: Single channel array with class indices (H x W)
        label_values: RGB values for each class

    Returns:
        Color coded segmentation (H x W x 3)
    """
    colour_codes = np.array(label_values)
    return colour_codes[image.astype(int)]


def visualize(**images):
    """
    Plot images in one row.

    Args:
        **images: Named images to visualize
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.tight_layout()
    plt.show()


def pad_to_multiple(img_np: np.ndarray, k: int = 32, mode: str = "reflect") -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image to be divisible by k (useful for models requiring specific input sizes).

    Args:
        img_np: Input image (H x W x C)
        k: Multiple to pad to
        mode: Padding mode ('reflect', 'constant', 'edge')

    Returns:
        Tuple of (padded_image, pad_info) where pad_info is (top, left, original_H, original_W)
    """
    H, W = img_np.shape[:2]
    Hn = int(np.ceil(H / k) * k)
    Wn = int(np.ceil(W / k) * k)
    pad_h, pad_w = Hn - H, Wn - W

    if pad_h == 0 and pad_w == 0:
        return img_np, (0, 0, H, W)

    # Calculate padding for each side
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Handle different numbers of dimensions
    if len(img_np.shape) == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))

    if mode == "reflect":
        padded = np.pad(img_np, pad_width, mode="reflect")
    elif mode == "edge":
        padded = np.pad(img_np, pad_width, mode="edge")
    else:
        padded = np.pad(img_np, pad_width, mode="constant", constant_values=0)

    return padded, (top, left, H, W)


def crop_from_pad(arr: np.ndarray, pad_info: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop array back to original size after padding.

    Args:
        arr: Array to crop (can be CHW or HW format)
        pad_info: Tuple of (top, left, original_H, original_W)

    Returns:
        Cropped array
    """
    top, left, H, W = pad_info

    if len(arr.shape) == 3:
        if arr.shape[0] < arr.shape[-1]:  # CHW format
            return arr[:, top:top + H, left:left + W]
        else:  # HWC format
            return arr[top:top + H, left:left + W, :]
    else:  # HW format
        return arr[top:top + H, left:left + W]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    filepath: str,
    **kwargs
):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        filepath: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', None),
        'val_loss': checkpoint.get('val_loss', None),
    }

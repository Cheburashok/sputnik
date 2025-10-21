"""
Evaluation and testing functionality for land segmentation.
Provides metrics computation and visualization of results.
"""
import os
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import reverse_one_hot, colour_code_segmentation, visualize


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        class_names: Optional list of class names

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)

    total_samples = 0
    results = {
        'predictions': [],
        'ground_truth': [],
    }

    for images, masks in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)

        # Get predictions
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        # Store results
        results['predictions'].extend(probs.cpu().numpy())
        results['ground_truth'].extend(masks.cpu().numpy())
        total_samples += images.size(0)

    print(f"Evaluated {total_samples} samples")
    return results


@torch.no_grad()
def visualize_predictions(
    model: torch.nn.Module,
    dataset,
    class_rgb_values: List[List[int]],
    device: str = 'cuda',
    num_samples: int = 5,
    save_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
):
    """
    Visualize model predictions on random samples.

    Args:
        model: PyTorch model
        dataset: Dataset to sample from (should return image, mask)
        class_rgb_values: RGB values for each class
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save visualizations
        class_names: Optional list of class names for titles
    """
    model.eval()
    model.to(device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        # Get data
        image_tensor, gt_mask = dataset[idx]

        # Prepare for visualization (need version without preprocessing)
        # This assumes dataset has a way to get non-preprocessed version
        # For now, we'll work with what we have

        # Get prediction
        x_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
        pred_logits = model(x_tensor)
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_mask = pred_probs.squeeze(0).cpu().numpy()

        # Convert masks from CHW to HWC for visualization
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        gt_mask = np.transpose(gt_mask, (1, 2, 0))

        # Get class predictions
        pred_classes = reverse_one_hot(pred_mask)
        gt_classes = reverse_one_hot(gt_mask)

        # Color code
        pred_colored = colour_code_segmentation(pred_classes, class_rgb_values)
        gt_colored = colour_code_segmentation(gt_classes, class_rgb_values)

        # Visualize
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(gt_colored)
        plt.title('Ground Truth', fontsize=14)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_colored)
        plt.title('Prediction', fontsize=14)
        plt.axis('off')

        plt.tight_layout()

        if save_dir:
            save_path = os.path.join(save_dir, f'prediction_{i}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()

        plt.close()


@torch.no_grad()
def test_and_visualize(
    model: torch.nn.Module,
    test_dataset,
    test_dataset_vis,
    class_rgb_values: List[List[int]],
    device: str = 'cuda',
    save_dir: str = 'sample_predictions',
    class_names: Optional[List[str]] = None,
):
    """
    Test model and create comprehensive visualizations.

    This mimics the original notebook testing functionality.

    Args:
        model: PyTorch model
        test_dataset: Dataset with preprocessing applied
        test_dataset_vis: Dataset without preprocessing (for visualization)
        class_rgb_values: RGB values for each class
        device: Device to run inference on
        save_dir: Directory to save predictions
        class_names: Optional list of class names
    """
    model.eval()
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        class_names = [f'class_{i}' for i in range(len(class_rgb_values))]

    for idx in tqdm(range(len(test_dataset)), desc="Testing"):
        # Get preprocessed image and ground truth
        image, gt_mask = test_dataset[idx]
        image_vis = test_dataset_vis[idx][0].astype('uint8')

        # Predict
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert from CHW to HWC
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        gt_mask = np.transpose(gt_mask, (1, 2, 0))

        # Get class-specific heatmap (e.g., urban land)
        pred_urban_heatmap = None
        if 'urban_land' in class_names:
            urban_idx = class_names.index('urban_land')
            pred_urban_heatmap = pred_mask[:, :, urban_idx]

        # Color code segmentation
        pred_mask_colored = colour_code_segmentation(
            reverse_one_hot(pred_mask),
            class_rgb_values
        )
        gt_mask_colored = colour_code_segmentation(
            reverse_one_hot(gt_mask),
            class_rgb_values
        )

        # Save concatenated image
        comparison = np.hstack([image_vis, gt_mask_colored, pred_mask_colored])
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_dir, f"sample_pred_{idx}.png")
        cv2.imwrite(save_path, comparison_bgr)

        # Visualize
        vis_dict = {
            'original_image': image_vis,
            'ground_truth_mask': gt_mask_colored,
            'predicted_mask': pred_mask_colored,
        }

        if pred_urban_heatmap is not None:
            vis_dict['pred_urban_land_heatmap'] = pred_urban_heatmap

        visualize(**vis_dict)

    print(f"Saved all predictions to {save_dir}")


def compute_iou(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    num_classes: int,
    eps: float = 1e-7
) -> dict:
    """
    Compute Intersection over Union (IoU) for each class.

    Args:
        predictions: Predicted masks (N x H x W) or (N x C x H x W)
        ground_truth: Ground truth masks (N x H x W) or (N x C x H x W)
        num_classes: Number of classes
        eps: Small epsilon for numerical stability

    Returns:
        Dictionary with per-class IoU and mean IoU
    """
    ious = []

    for cls in range(num_classes):
        if predictions.ndim == 4:  # One-hot format
            pred_cls = predictions[:, cls]
            gt_cls = ground_truth[:, cls]
        else:  # Class indices
            pred_cls = (predictions == cls)
            gt_cls = (ground_truth == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        iou = (intersection + eps) / (union + eps)
        ious.append(iou)

    return {
        'per_class_iou': ious,
        'mean_iou': np.mean(ious)
    }

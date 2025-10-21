"""
Inference module for land segmentation.
Provides clean, Pythonic inference using proper Dataset patterns.
This replaces the unpythonic test_folder() function.
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

from .datasets import InferenceDataset
from .utils import crop_from_pad, visualize
from .config import InferenceConfig


class Predictor:
    """
    Predictor class for land segmentation inference.

    This is the Pythonic replacement for test_folder().
    Uses proper PyTorch Dataset and follows clean coding patterns.

    Args:
        model: PyTorch segmentation model
        preprocessing_fn: Preprocessing function (e.g., from SMP encoder)
        config: Inference configuration
        class_names: List of class names
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocessing_fn: Optional[callable] = None,
        config: Optional[InferenceConfig] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.preprocessing_fn = preprocessing_fn
        self.config = config or InferenceConfig()
        self.class_names = class_names or ['class_0', 'class_1', 'class_2']

        self.model.eval()
        self.model.to(self.config.device)

    @torch.no_grad()
    def predict_folder(
        self,
        folder_path: Union[str, Path],
        save_dir: Optional[Union[str, Path]] = None,
        visualize_results: bool = True,
        return_predictions: bool = False,
    ) -> Optional[List[dict]]:
        """
        Predict on all images in a folder.

        Args:
            folder_path: Path to folder containing images
            save_dir: Optional directory to save results
            visualize_results: Whether to visualize predictions
            return_predictions: Whether to return prediction results

        Returns:
            Optional list of prediction dictionaries if return_predictions=True
        """
        # Create dataset
        dataset = InferenceDataset(
            image_folder=folder_path,
            preprocessing_fn=self.preprocessing_fn,
            pad_multiple=self.config.pad_multiple,
            pad_mode=self.config.pad_mode,
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0  # Windows-friendly
        )

        predictions = []

        # Iterate through images
        for batch_tensors, batch_metadata in tqdm(dataloader, desc="Predicting"):
            batch_tensors = batch_tensors.to(self.config.device)

            # Predict
            logits = self.model(batch_tensors)

            # Process each image in batch
            for i in range(len(batch_tensors)):
                pred_result = self._process_prediction(
                    logits[i],
                    batch_metadata,
                    i,
                    visualize=visualize_results
                )

                if save_dir:
                    self._save_prediction(pred_result, save_dir)

                if return_predictions:
                    predictions.append(pred_result)

        if return_predictions:
            return predictions

    @torch.no_grad()
    def predict_single(
        self,
        image_path: Union[str, Path],
        visualize_result: bool = True,
    ) -> dict:
        """
        Predict on a single image.

        Args:
            image_path: Path to image
            visualize_result: Whether to visualize the result

        Returns:
            Dictionary with prediction results
        """
        # Create dataset with single image
        folder_path = Path(image_path).parent
        dataset = InferenceDataset(
            image_folder=folder_path,
            preprocessing_fn=self.preprocessing_fn,
            pad_multiple=self.config.pad_multiple,
            pad_mode=self.config.pad_mode,
        )

        # Find the specific image
        image_name = Path(image_path).name
        idx = None
        for i, path in enumerate(dataset.image_paths):
            if path.name == image_name:
                idx = i
                break

        if idx is None:
            raise ValueError(f"Image {image_name} not found in {folder_path}")

        # Get tensor and metadata
        tensor, metadata = dataset[idx]
        tensor = tensor.unsqueeze(0).to(self.config.device)

        # Predict
        logits = self.model(tensor)
        pred_result = self._process_prediction(logits[0], metadata, 0, visualize=visualize_result)

        return pred_result

    def _process_prediction(
        self,
        logits: torch.Tensor,
        metadata: dict,
        batch_idx: int,
        visualize: bool = True,
    ) -> dict:
        """
        Process model logits into final predictions.

        Args:
            logits: Model output logits (C x H x W)
            metadata: Metadata dictionary from dataset
            batch_idx: Index in batch
            visualize: Whether to visualize

        Returns:
            Dictionary with prediction results
        """
        # Get metadata for this specific image
        if isinstance(metadata['path'], list):
            path = metadata['path'][batch_idx]
            pad_info = metadata['pad_info'][batch_idx]
            original_image = metadata['original_image'][batch_idx]
        else:
            path = metadata['path']
            pad_info = metadata['pad_info']
            original_image = metadata['original_image']

        # Compute probabilities
        probs = torch.softmax(logits, dim=0).cpu().numpy()  # C x H x W

        # Check for "other" class (last class)
        if len(probs) > len(self.class_names):
            other_class_pred = probs[-1]
            other_ratio = other_class_pred.sum() / other_class_pred.size
            if other_ratio > self.config.other_class_ratio_threshold:
                print(f"Warning: OTHER class ratio = {other_ratio:.3f} for {Path(path).name}")

        # Crop back to original size
        probs = crop_from_pad(probs, pad_info)

        # Get class predictions
        pred_idx = np.argmax(probs, axis=0)  # H x W

        # Create class-specific maps
        class_maps = {}
        for cid, cname in enumerate(self.class_names):
            if cid < len(probs):
                if self.config.show_soft_predictions:
                    class_maps[cname] = probs[cid].astype(np.float32)
                else:
                    class_maps[cname] = (pred_idx == cid).astype(np.uint8) * 255

        # Visualize if requested
        if visualize:
            vis_dict = {'original': original_image}
            vis_dict.update(class_maps)
            visualize(**vis_dict)

        return {
            'path': path,
            'probabilities': probs,
            'predictions': pred_idx,
            'class_maps': class_maps,
            'original_image': original_image,
        }

    def _save_prediction(self, pred_result: dict, save_dir: Union[str, Path]):
        """Save prediction results to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(pred_result['path']).stem

        # Save class maps
        for class_name, class_map in pred_result['class_maps'].items():
            save_path = save_dir / f"{image_name}_{class_name}.png"
            if class_map.dtype == np.float32:
                # Save as normalized image
                import cv2
                cv2.imwrite(str(save_path), (class_map * 255).astype(np.uint8))
            else:
                import cv2
                cv2.imwrite(str(save_path), class_map)

    @torch.no_grad()
    def compute_timeseries_statistics(
        self,
        folder_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        window: int = 12,
        title: Optional[str] = None,
    ) -> dict:
        """
        Compute and plot class distribution statistics over time.

        This replaces the get_stat_timeseries function.

        Args:
            folder_path: Path to folder with chronologically named images
            save_path: Optional path to save results (CSV and plot)
            window: Moving average window size
            title: Optional plot title

        Returns:
            Dictionary with timeseries data
        """
        # Create dataset
        dataset = InferenceDataset(
            image_folder=folder_path,
            preprocessing_fn=self.preprocessing_fn,
            pad_multiple=self.config.pad_multiple,
            pad_mode=self.config.pad_mode,
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        names = []
        class_proportions = []

        # Process each image
        for tensor, metadata in tqdm(dataloader, desc="Computing statistics"):
            tensor = tensor.to(self.config.device)

            # Predict
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)  # B x C x H x W

            # Compute spatial mean to get class proportions
            probs_mean = probs.mean(dim=(2, 3))  # B x C
            probs_mean = probs_mean / (probs_mean.sum(dim=1, keepdim=True) + 1e-8)

            class_proportions.append(probs_mean[0].cpu().numpy())

            # Extract timestamp from filename
            img_name = Path(metadata['path'][0]).stem
            names.append(img_name[:10])  # First 10 chars as timestamp

        # Convert to array
        series = np.array(class_proportions)  # N x C
        N, C = series.shape

        # Create dataframe
        df = pd.DataFrame(series, columns=self.class_names[:C])
        df.insert(0, 'image', names)

        # Compute moving averages if requested
        if window > 1:
            for col in self.class_names[:C]:
                ma_col = f"{col}_MA{window}"
                df[ma_col] = df[col].rolling(window=window, min_periods=1).mean()

        # Plot
        self._plot_timeseries(df, series, names, window, title, save_path)

        # Save CSV
        if save_path:
            csv_path = Path(save_path).with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved CSV: {csv_path}")

        return {
            'names': names,
            'series': series,
            'class_names': self.class_names[:C],
            'dataframe': df,
        }

    def _plot_timeseries(
        self,
        df: pd.DataFrame,
        series: np.ndarray,
        names: List[str],
        window: int,
        title: Optional[str],
        save_path: Optional[Union[str, Path]],
    ):
        """Plot timeseries statistics."""
        N, C = series.shape
        x = np.arange(N)

        # Use moving average if window > 1
        if window > 1:
            plot_series = []
            for col in self.class_names[:C]:
                ma_col = f"{col}_MA{window}"
                plot_series.append(df[ma_col].values)
            plot_series = np.array(plot_series)
        else:
            plot_series = series.T

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 5), dpi=120)

        colors = plt.cm.Set2(np.linspace(0, 1, 7))
        ax.stackplot(x, plot_series, labels=self.class_names[:C], colors=colors[:C])

        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            fancybox=True,
            shadow=True,
            ncol=min(C, 5)
        )

        ax.set_title(title or "Class distribution over time")
        ax.set_xlabel("Image index (chronological)")
        ax.set_ylabel("Proportion (sums to 1)")

        # Set x-ticks
        if N > 20:
            step = max(1, N // 20)
            ax.set_xticks(x[::step])
            ax.set_xticklabels([names[i] for i in range(0, N, step)], rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')

        plt.tight_layout()

        # Save or show
        if save_path:
            plot_path = Path(save_path).with_suffix('.png')
            plt.savefig(plot_path, bbox_inches='tight', dpi=150)
            print(f"Saved plot: {plot_path}")
            plt.close()
        else:
            plt.show()


def create_predictor(
    model_path: Union[str, Path],
    encoder: str = 'resnet50',
    encoder_weights: str = 'imagenet',
    class_names: Optional[List[str]] = None,
    config: Optional[InferenceConfig] = None,
) -> Predictor:
    """
    Create a Predictor instance from a saved model.

    Args:
        model_path: Path to saved model checkpoint
        encoder: Encoder name
        encoder_weights: Encoder weights
        class_names: List of class names
        config: Inference configuration

    Returns:
        Predictor instance
    """
    import segmentation_models_pytorch as smp

    # Load model
    model = torch.load(model_path, map_location='cpu')

    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    # Create predictor
    predictor = Predictor(
        model=model,
        preprocessing_fn=preprocessing_fn,
        config=config,
        class_names=class_names,
    )

    return predictor


if __name__ == '__main__':
    # Example usage
    predictor = create_predictor(
        model_path='best_model_kaggle.pth',
        class_names=['urban_land', 'agriculture_land', 'forest_land']
    )

    # Predict on folder
    predictor.predict_folder(
        folder_path='data/test_monuments',
        save_dir='predictions',
        visualize_results=True
    )

    # Compute timeseries statistics
    stats = predictor.compute_timeseries_statistics(
        folder_path='data/test_monuments',
        save_path='timeseries_results',
        window=12
    )

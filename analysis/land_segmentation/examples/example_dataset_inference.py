"""
Example showing how to use InferenceDataset directly with DataLoader.
This demonstrates the proper PyTorch pattern for inference.
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.land_segmentation import InferenceDataset
from analysis.land_segmentation.utils import crop_from_pad


def main():
    """Main inference function using InferenceDataset."""

    # Configuration
    model_path = 'checkpoints/best_model.pth'
    image_folder = 'data/collections/Zvarivank/images'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("Loading model...")
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)

    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')

    # Create InferenceDataset - This is the Pythonic way!
    print(f"\nCreating dataset from: {image_folder}")
    dataset = InferenceDataset(
        image_folder=image_folder,
        preprocessing_fn=preprocessing_fn,
        file_extensions=('*.png',),
        pad_multiple=32,
        pad_mode='reflect'
    )

    print(f"Found {len(dataset)} images")

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0  # Windows-friendly
    )

    # Inference loop
    all_predictions = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, (batch_tensors, batch_metadata) in enumerate(dataloader):
            # Move to device
            batch_tensors = batch_tensors.to(device)

            # Inference
            logits = model(batch_tensors)
            probs = torch.softmax(logits, dim=1)

            # Process each image in batch
            for i in range(len(batch_tensors)):
                # Get metadata
                img_path = batch_metadata['path'][i]
                pad_info = batch_metadata['pad_info'][i]
                original_shape = batch_metadata['original_shape'][i]

                # Get probabilities and crop back to original size
                prob_map = probs[i].cpu().numpy()  # C x H x W
                prob_map = crop_from_pad(prob_map, pad_info)

                # Get class predictions
                pred_classes = np.argmax(prob_map, axis=0)  # H x W

                # Store result
                result = {
                    'path': img_path,
                    'predictions': pred_classes,
                    'probabilities': prob_map,
                    'shape': pred_classes.shape
                }
                all_predictions.append(result)

                print(f"  Processed: {Path(img_path).name} -> shape {pred_classes.shape}")

    print(f"\nTotal predictions: {len(all_predictions)}")

    # Example: Access specific prediction
    if all_predictions:
        first_pred = all_predictions[0]
        print(f"\nFirst prediction:")
        print(f"  Path: {first_pred['path']}")
        print(f"  Shape: {first_pred['shape']}")
        print(f"  Unique classes: {np.unique(first_pred['predictions'])}")


if __name__ == '__main__':
    main()

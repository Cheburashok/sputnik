"""
Example inference script using the Predictor class.
This demonstrates the Pythonic way to do inference (replacing test_folder).
"""
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.land_segmentation import create_predictor, InferenceConfig


def main():
    """Main inference function."""

    # Configuration
    model_path = 'checkpoints/best_model.pth'
    image_folder = 'data/collections/Zvarivank/images'
    output_dir = 'predictions'

    # Class names for your model
    class_names = [
        'urban_land',
        'agriculture_land',
        'rangeland',
        'forest_land',
        'water',
        'barren_land',
        'unknown'
    ]

    # Create inference config
    config = InferenceConfig(
        device='cuda',
        batch_size=1,
        pad_multiple=32,
        pad_mode='reflect',
        show_soft_predictions=False,
        save_visualizations=True
    )

    # Create predictor
    print("Loading model...")
    predictor = create_predictor(
        model_path=model_path,
        encoder='resnet50',
        encoder_weights='imagenet',
        class_names=class_names,
        config=config
    )

    # Option 1: Predict on entire folder
    print(f"\nPredicting on images in: {image_folder}")
    predictor.predict_folder(
        folder_path=image_folder,
        save_dir=output_dir,
        visualize_results=False,  # Set to True to show plots
        return_predictions=False
    )

    print(f"\nPredictions saved to: {output_dir}")

    # Option 2: Predict on single image
    # single_image = Path(image_folder) / 'your_image.png'
    # if single_image.exists():
    #     print(f"\nPredicting on single image: {single_image}")
    #     result = predictor.predict_single(
    #         image_path=single_image,
    #         visualize_result=True
    #     )
    #     print(f"Prediction shape: {result['predictions'].shape}")


if __name__ == '__main__':
    main()

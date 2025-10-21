"""
Example script for computing time-series statistics.
This demonstrates how to analyze land cover changes over time.
"""
import sys
from pathlib import Path

# Add parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.land_segmentation import create_predictor, InferenceConfig


def main():
    """Main time-series analysis function."""

    # Configuration
    model_path = 'checkpoints/best_model.pth'
    image_folder = 'data/collections/Zvarivank/images'
    output_path = 'timeseries_results/zvarivank'

    # Class names
    class_names = [
        'urban_land',
        'agriculture_land',
        'rangeland',
        'forest_land',
        'water',
        'barren_land',
        'unknown'
    ]

    # Create predictor
    print("Loading model...")
    predictor = create_predictor(
        model_path=model_path,
        encoder='resnet50',
        encoder_weights='imagenet',
        class_names=class_names
    )

    # Compute time-series statistics
    print(f"\nComputing time-series statistics for: {image_folder}")
    print("This will process all images and compute class proportions...")

    stats = predictor.compute_timeseries_statistics(
        folder_path=image_folder,
        save_path=output_path,
        window=12,  # 12-image moving average
        title='Zvarivank Land Cover Changes Over Time'
    )

    print(f"\nResults saved to:")
    print(f"  - CSV: {output_path}.csv")
    print(f"  - Plot: {output_path}.png")

    # Print summary statistics
    print("\nSummary:")
    print(f"Total images processed: {len(stats['names'])}")
    print(f"Classes: {', '.join(stats['class_names'])}")
    print("\nMean class proportions:")
    for i, class_name in enumerate(stats['class_names']):
        mean_prop = stats['series'][:, i].mean()
        print(f"  {class_name}: {mean_prop:.2%}")


if __name__ == '__main__':
    main()

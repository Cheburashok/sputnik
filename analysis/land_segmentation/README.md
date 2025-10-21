# Land Segmentation Package

Implementation for training and inference of land cover segmentation models on satellite imagery.

## Overview

This package provides a well-structured codebase for semantic segmentation of satellite images, particularly focused on land cover classification (urban, agriculture, forest, water, etc.).

## Project Structure

```
land_segmentation/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclasses
├── datasets.py              # PyTorch Dataset implementations
├── transforms.py            # Data augmentation pipelines
├── utils.py                 # Helper functions
├── train.py                 # Training logic
├── evaluate.py              # Evaluation and testing
├── predict.py               # Inference with Predictor class
├── requirements.txt         # Dependencies
├── README.md                # This file
└── archive/                 # Original notebooks and scripts
    ├── train.ipynb
    ├── test.ipynb
    └── predict_helpers.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training a Model

```python
from land_segmentation import ModelConfig, DataConfig, TrainingConfig
from land_segmentation.train import train

# Configure training
model_config = ModelConfig(
    encoder='resnet50',
    encoder_weights='imagenet',
    activation='sigmoid'
)

data_config = DataConfig(
    data_dir='/path/to/dataset',
    metadata_csv='/path/to/metadata.csv',
    train_img_size=1024,
)

training_config = TrainingConfig(
    batch_size=12,
    num_epochs=40,
    learning_rate=0.00008,
    checkpoint_dir='checkpoints'
)

# Train model
train(model_config, data_config, training_config)
```

### 2. Inference on Images (The Pythonic Way)

**Old approach (unpythonic):**
```python
# DON'T DO THIS - this was the old way
test_folder(folder_path, model, preprocessing_fn)
```

**New approach (Pythonic):**
```python
from land_segmentation import create_predictor, InferenceConfig

# Create predictor
predictor = create_predictor(
    model_path='best_model.pth',
    class_names=['urban_land', 'agriculture_land', 'forest_land']
)

# Predict on entire folder
predictor.predict_folder(
    folder_path='data/test_images',
    save_dir='predictions',
    visualize_results=True
)

# Predict on single image
result = predictor.predict_single(
    image_path='data/test_images/image_001.png',
    visualize_result=True
)
```

### 3. Using InferenceDataset Directly

```python
from torch.utils.data import DataLoader
from land_segmentation import InferenceDataset
import segmentation_models_pytorch as smp

# Load model
model = torch.load('best_model.pth')
model.eval()

# Get preprocessing function
preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')

# Create dataset - this is the proper PyTorch way!
dataset = InferenceDataset(
    image_folder='data/test_images',
    preprocessing_fn=preprocessing_fn,
    pad_multiple=32
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# Inference loop
for batch_tensors, batch_metadata in dataloader:
    batch_tensors = batch_tensors.to('cuda')
    logits = model(batch_tensors)
    # Process predictions...
```

### 4. Time-Series Analysis

```python
# Compute land cover statistics over time
stats = predictor.compute_timeseries_statistics(
    folder_path='data/chronological_images',
    save_path='timeseries_results',
    window=12,  # 12-image moving average
    title='Land Cover Changes Over Time'
)

# Returns dictionary with:
# - 'series': numpy array of class proportions (N x C)
# - 'names': list of image names/timestamps
# - 'dataframe': pandas DataFrame with raw and smoothed data
```

### 5. Evaluation and Testing

```python
from land_segmentation.evaluate import test_and_visualize, evaluate_model
from land_segmentation import LandCoverDataset

# Create test datasets
test_dataset = LandCoverDataset(
    test_df,
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=class_rgb_values
)

test_dataset_vis = LandCoverDataset(
    test_df,
    class_rgb_values=class_rgb_values  # No preprocessing for vis
)

# Test and visualize
test_and_visualize(
    model=model,
    test_dataset=test_dataset,
    test_dataset_vis=test_dataset_vis,
    class_rgb_values=class_rgb_values,
    save_dir='test_results'
)
```

## Configuration

All configurations are managed through dataclasses in `config.py`:

- `ModelConfig`: Model architecture settings
- `DataConfig`: Dataset and data loading configuration
- `TrainingConfig`: Training hyperparameters
- `InferenceConfig`: Inference settings

Example:
```python
from land_segmentation import InferenceConfig

config = InferenceConfig(
    device='cuda',
    batch_size=4,
    pad_multiple=32,
    pad_mode='reflect',
    show_soft_predictions=False,
    other_class_ratio_threshold=0.3
)
```

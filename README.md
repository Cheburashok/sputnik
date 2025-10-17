# Sputnik - Copernicus Satellite Imagery Downloader

A Python library for downloading and processing satellite imagery from the Copernicus Data Space Ecosystem. Supports Sentinel-2 and Landsat-7 data with automatic authentication, quality filtering, and streaming downloads.

## Features

- 🛰️ **Multiple Satellites**: Sentinel-2, Sentinel-3, Landsat-7
- 🎯 **Asset Types**: Full PRODUCT archives or QUICKLOOK previews
- 🔐 **Automatic Authentication**: Fresh OAuth2 tokens from credentials
- 📦 **Memory Efficient**: Streaming downloads for large files (800MB+)
- 🌤️ **Quality Filtering**: Cloud cover thresholds
- 🖼️ **Image Processing**: Extract True Color Images (TCI) from SAFE archives
- 🗂️ **Organized Storage**: Automatic directory structure
- ⏸️ **Resume Support**: Skip already-downloaded files

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Quick Start

### 1. Set Up Credentials

Create a `.credentials.yaml` file in the project root:

```yaml
username: your.email@example.com
password: YourPassword123
```

> **Security Note**: Add `.credentials.yaml` to `.gitignore` (already included)

### 2. Prepare Monument Data

Create a CSV file with monument locations (`data/monuments.csv`):

```csv
name,min_lon,min_lat,max_lon,max_lat
Taj Mahal,78.0400,27.1700,78.0450,27.1750
Eiffel Tower,2.2920,48.8574,2.2970,48.8594
```

> **Tip**: Find bounding boxes using [bboxfinder.com](http://bboxfinder.com/)

### 3. Download Satellite Imagery

```python
from copernicus_utils.manager import MonumentImageManager
import datetime as dt

# Initialize manager
manager = MonumentImageManager(
    "data/monuments.csv",
    collection="SENTINEL-2",       # or "LANDSAT-7"
    asset_type="PRODUCT",          # or "QUICKLOOK"
    max_cloud_cover=30.0,          # Maximum 30% cloud cover
    lookback_days=30               # Search last 30 days
)

# Download time series for all monuments
results = manager.collect_time_series_for_all_monuments(
    start_date=dt.date(2025, 6, 1),
    end_date=dt.date(2025, 6, 30)
)

# Print results
for monument_name, paths in results.items():
    print(f"{monument_name}: {len(paths)} images")
```

## API Reference

### Types

```python
from copernicus_utils.copernicus_api import AssetType, Collection

AssetType = Literal["PRODUCT", "QUICKLOOK"]
Collection = Literal["SENTINEL-2", "LANDSAT-7"]
```

### Core Functions

#### `fetch_latest_image_for_bbox()`

Get the most recent high-quality image for a location:

```python
from copernicus_utils.copernicus_api import fetch_latest_image_for_bbox
import datetime as dt

image = fetch_latest_image_for_bbox(
    bbox=(lon_min, lat_min, lon_max, lat_max),
    target_date=dt.datetime.now(),
    lookback_days=30,
    max_cloud_cover=20.0,
    collection="SENTINEL-2",
    asset_type="PRODUCT"
)

# Save to disk
with open("image.zip", "wb") as f:
    f.write(image.content)
```

#### `fetch_all_images_in_period()`

Generator that yields download info for all images in a date range:

```python
from copernicus_utils.copernicus_api import (
    fetch_all_images_in_period,
    _download_asset_to_file,
    _read_token
)

# Stream downloads one at a time
for item in fetch_all_images_in_period(
    bbox=(lon_min, lat_min, lon_max, lat_max),
    start_date=dt.date(2025, 6, 1),
    end_date=dt.date(2025, 6, 30),
    max_cloud_cover=30.0,
    collection="SENTINEL-2",
    asset_type="PRODUCT"
):
    # Build output path
    output_path = Path(f"downloads/{item['index']}.zip")

    # Stream download to disk (memory-efficient)
    token = _read_token(item['token_path'])
    _download_asset_to_file(item['href'], token, output_path)

    print(f"Downloaded {item['index']}/{item['total']}: {item['metadata']['timestamp']}")
```

### Monument Image Manager

#### Initialize

```python
from copernicus_utils.manager import MonumentImageManager

manager = MonumentImageManager(
    monuments_csv="data/monuments.csv",     # Path to CSV
    storage_root="data/collections",        # Storage directory
    collection="SENTINEL-2",                # Satellite collection
    asset_type="PRODUCT",                   # Asset type
    max_cloud_cover=30.0,                   # Cloud cover threshold (%)
    lookback_days=30,                       # Search window (days)
    token_path=None                         # Optional: custom token path
)
```

#### Download Methods

```python
# Download for a single monument
monument = manager.monuments[0]
paths = manager.collect_time_series_for_monument(
    monument,
    start_date=dt.date(2025, 6, 1),
    end_date=dt.date(2025, 6, 30)
)

# Download for all monuments
results = manager.collect_time_series_for_all_monuments(
    start_date=dt.date(2025, 6, 1),
    end_date=dt.date(2025, 6, 30)
)
```

#### Extract Preview Images

Extract True Color Images (TCI) from SAFE archives and convert to PNG:

```python
# Extract previews for a single monument
monument = manager.monuments[0]
png_files = manager.extract_previews_for_monument(
    monument,
    max_size_mb=10.0,  # Maximum PNG size in MB
    quality=85         # PNG quality (0-100)
)

# Extract previews for all monuments
previews = manager.extract_previews_for_all_monuments(
    max_size_mb=10.0,
    quality=85
)
```

## Directory Structure

### PRODUCT Assets

```
data/collections/
└── Monument_Name/
    ├── images/                    # Full SAFE archives
    │   ├── S2A_MSIL1C_...zip     # ~800MB each
    │   └── ...
    ├── metadata/                  # Image metadata
    │   ├── *.json
    │   └── ...
    └── previews/                  # Extracted TCI images
        ├── *.png                  # <10MB each
        └── ...
```

### QUICKLOOK Assets

```
data/collections/
└── Monument_Name/
    ├── quicklook/                 # Preview images
    │   ├── *.jpg                  # ~1MB each
    │   └── ...
    └── quicklook_metadata/        # Preview metadata
        ├── *.json
        └── ...
```

## Advanced Usage

### Custom Asset Selection

```python
# Download full products (default)
manager_product = MonumentImageManager(
    "data/monuments.csv",
    asset_type="PRODUCT"  # 800MB+ SAFE archives
)

# Download quicklook previews
manager_quicklook = MonumentImageManager(
    "data/monuments.csv",
    asset_type="QUICKLOOK"  # ~1MB JPEG previews
)
```

### Multiple Satellite Collections

```python
# Sentinel-2 (10m resolution, RGB)
manager_s2 = MonumentImageManager(
    "data/monuments.csv",
    collection="SENTINEL-2"
)

# Landsat-7 (30m resolution)
manager_l7 = MonumentImageManager(
    "data/monuments.csv",
    collection="LANDSAT-7"
)
```

### Quality Control

```python
# Strict quality requirements
manager = MonumentImageManager(
    "data/monuments.csv",
    max_cloud_cover=10.0,  # Only very clear images
    lookback_days=60       # Longer search window
)

# Relaxed quality requirements
manager = MonumentImageManager(
    "data/monuments.csv",
    max_cloud_cover=50.0,  # Accept cloudier images
    lookback_days=7        # Recent images only
)
```

### Extract and Process Images

```python
from copernicus_utils.image_processing import extract_tci_from_safe
from pathlib import Path

# Extract TCI from a single SAFE archive
png_path = extract_tci_from_safe(
    safe_zip_path=Path("data/S2A_MSIL1C_...zip"),
    output_path=Path("output/preview.png"),
    max_size_mb=5.0,  # Downscale to 5MB max
    quality=85        # PNG quality
)

# Batch extract from multiple archives
from copernicus_utils.image_processing import batch_extract_tcis

safe_archives = list(Path("data").glob("*.zip"))
png_files = batch_extract_tcis(
    safe_archives,
    output_dir=Path("output/previews"),
    max_size_mb=10.0,
    quality=85
)
```

## Authentication

### Automatic Token Generation

The library automatically generates fresh OAuth2 tokens from your credentials:

1. Reads `.credentials.yaml`
2. Requests token from Copernicus Identity Service
3. Token valid for 10 minutes
4. Fresh token generated for each download batch

### Manual Token (Alternative)

```python
# Create .copernicus_token.txt with your token
# Token expires after 10 minutes

manager = MonumentImageManager(
    "data/monuments.csv",
    token_path=Path(".copernicus_token.txt")
)
```

## Memory Management

### Streaming Downloads (PRODUCT)

Large files (800MB+) are streamed in 8KB chunks:

```python
# Automatic for PRODUCT assets
manager = MonumentImageManager(
    "data/monuments.csv",
    asset_type="PRODUCT"  # Uses streaming
)

# Memory usage: ~8KB constant
```

### In-Memory Downloads (QUICKLOOK)

Small files (~1MB) are downloaded to memory:

```python
# Automatic for QUICKLOOK assets
manager = MonumentImageManager(
    "data/monuments.csv",
    asset_type="QUICKLOOK"  # Uses in-memory
)

# Memory usage: ~1-2MB per file (temporary)
```

## Error Handling

The library handles common errors gracefully:

- **401 Unauthorized**: Automatically generates fresh tokens
- **403 Forbidden**: Handles redirects with auth headers
- **No images found**: Logs warning and continues
- **Download failures**: Logs error and continues with next image

```python
# Errors are logged, not raised
import logging

logging.basicConfig(level=logging.INFO)

manager = MonumentImageManager("data/monuments.csv")
results = manager.collect_time_series_for_all_monuments(
    start_date=dt.date(2025, 6, 1),
    end_date=dt.date(2025, 6, 30)
)

# Check results
for monument, paths in results.items():
    if not paths:
        print(f"Warning: No images found for {monument}")
```

## Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Run downloads with logging
manager = MonumentImageManager("data/monuments.csv")
manager.collect_time_series_for_all_monuments(...)

# Output:
# INFO: Collecting PRODUCT time series for 'Taj Mahal' from 2025-06-01 to 2025-06-30
# INFO: Searching for all images in bbox (78.04, 27.17, 78.045, 27.175) from 2025-06-01 to 2025-06-30 (30 days, max_cloud_cover=30.0%)
# INFO: Found 5 suitable images (out of 12 total)
# INFO:   Downloading 1/5: 2025-06-01T07:36:19.024000Z
# INFO: Streaming 823332290 bytes to S2A_MSIL1C_...zip
# ...
```

## FAQ

### Q: How do I get Copernicus credentials?

A: Register at [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) and use those credentials in `.credentials.yaml`.

### Q: Why do I get 401 errors?

A: Tokens expire after 10 minutes. The library auto-generates fresh tokens from `.credentials.yaml`. Ensure the file exists and credentials are correct.

### Q: What's the difference between PRODUCT and QUICKLOOK?

- **PRODUCT**: Full SAFE archive (~800MB), contains all bands and metadata
- **QUICKLOOK**: Preview image (~1MB JPEG), for quick visualization

### Q: How do I reduce download size?

Use QUICKLOOK instead of PRODUCT:
```python
manager = MonumentImageManager(
    "data/monuments.csv",
    asset_type="QUICKLOOK"  # ~1MB vs 800MB
)
```

### Q: Can I download Sentinel-3 or Sentinel-5P?

Currently supports:
- `SENTINEL-2` ✅
- `LANDSAT-7` ✅
- Others: Add to the `Collection` type and test

### Q: How do I extract RGB from SAFE archives?

```python
from copernicus_utils.image_processing import extract_tci_from_safe

png_path = extract_tci_from_safe(
    Path("data/S2A_MSIL1C_...zip"),
    Path("output/rgb.png")
)
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Update documentation for new features
4. Test with both SENTINEL-2 and LANDSAT-7

## License

See LICENSE file for details.

## Credits

- **Copernicus Data Space Ecosystem** - Satellite data provider
- **Sentinel-2** - ESA's high-resolution Earth observation mission
- **Landsat-7** - NASA/USGS Earth observation satellite

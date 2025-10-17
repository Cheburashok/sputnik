#!/usr/bin/env python3
"""
Crop a Sentinel-2 TCI_10m image to a WGS84 bbox.

Inputs:
  - tci_path: path to TCI_10m.jp2  (e.g. .../GRANULE/.../IMG_DATA/R10m/..._TCI_10m.jp2)
  - bbox_wgs84: (min_lon, min_lat, max_lon, max_lat)  in EPSG:4326
  - out_path: output file (e.g. crop.tif)

Notes:
  - Preserves georeferencing & CRS.
  - If your product is L2A/L1C without TCI, you’ll need to build RGB first.
"""

from pathlib import Path
from typing import Tuple, Optional
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
import numpy as np
from PIL import Image

def crop_tci_to_bbox(
    tci_path: str,
    bbox_wgs84: Tuple[float, float, float, float],
    out_path: str,
    driver: str = "GTiff",
    compress: str = "DEFLATE",
    overviews: bool = True,
    overview_levels=(2, 4, 8, 16),
):
    """
    Crop TCI_10m.jp2 to the exact bbox (in WGS84) and write out a georeferenced raster.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    with rasterio.open(tci_path) as src:
        # Reproject bbox to image CRS (usually UTM)
        bbox_dst = transform_bounds("EPSG:4326", src.crs, min_lon, min_lat, max_lon, max_lat, densify_pts=21)

        # Compute pixel window intersecting the bbox
        win = from_bounds(*bbox_dst, transform=src.transform)

        # Clip window to raster bounds (avoids out-of-range indexes)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

        if win.width <= 0 or win.height <= 0:
            raise ValueError("BBox does not intersect the TCI image.")

        # Read the crop
        data = src.read(window=win, boundless=False)

        # Derive new transform/width/height
        transform = rasterio.windows.transform(win, src.transform)
        height = int(win.height)
        width = int(win.width)

        # Prepare output profile
        profile = src.profile.copy()
        profile.update(
            driver=driver,
            height=height,
            width=width,
            transform=transform,
            compress=compress if driver in ("GTiff",) else None,
            tiled=True if driver == "GTiff" else None,
            BIGTIFF="IF_SAFER" if driver == "GTiff" else None,
        )

        # Write crop
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

            # Optional overviews for snappy viewers
            if overviews and driver == "GTiff":
                dst.build_overviews(overview_levels, Resampling.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")

    return Path(out_path)

def tif_to_png(
    tif_path: str,
    png_path: str,
    normalize: bool = True,
    bands: Optional[Tuple[int, int, int]] = None,
):
    """
    Convert a TIF file to PNG format.

    Args:
        tif_path: Path to input TIF file
        png_path: Path to output PNG file
        normalize: If True, normalize pixel values to 0-255 range
        bands: Tuple of (R, G, B) band indices (1-based). If None, uses first 3 bands or single band as grayscale

    Returns:
        Path object pointing to the created PNG file
    """
    with rasterio.open(tif_path) as src:
        # Determine which bands to read
        if bands is not None:
            # User specified bands (1-based indexing)
            data = np.array([src.read(b) for b in bands])
        elif src.count >= 3:
            # Read first 3 bands as RGB
            data = src.read([1, 2, 3])
        elif src.count == 1:
            # Single band - grayscale
            data = src.read(1)
        else:
            raise ValueError(f"Cannot convert {src.count} bands to PNG. Specify bands parameter.")

        # Convert to uint8 for PNG
        if normalize:
            # Normalize each band to 0-255
            if data.ndim == 3:  # Multi-band
                normalized = np.zeros_like(data, dtype=np.uint8)
                for i in range(data.shape[0]):
                    band = data[i]
                    min_val, max_val = band.min(), band.max()
                    if max_val > min_val:
                        normalized[i] = ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        normalized[i] = band.astype(np.uint8)
                data = normalized
            else:  # Single band
                min_val, max_val = data.min(), data.max()
                if max_val > min_val:
                    data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    data = data.astype(np.uint8)
        else:
            # Clip to 0-255 and convert
            data = np.clip(data, 0, 255).astype(np.uint8)

        # Create PIL Image and save as PNG
        if data.ndim == 3:
            # Transpose from (bands, height, width) to (height, width, bands)
            img = Image.fromarray(np.transpose(data, (1, 2, 0)), mode='RGB')
        else:
            # Grayscale
            img = Image.fromarray(data, mode='L')

        img.save(png_path, 'PNG')

    return Path(png_path)

if __name__ == "__main__":
    # --- EXAMPLE USAGE ---
    # 1) Set your inputs:
    TCI = r"C:\\Users\\dell\\OneDrive\\Desktop\\sputnik\\data\\collections\\Zvarivank\\images\\1\\S2B_MSIL1C_20250601T073619_N0511_R092_T38SPJ_20250601T092117.SAFE\\GRANULE\\L1C_T38SPJ_A043018_20250601T074250\\IMG_DATA\\T38SPJ_20250601T073619_TCI.jp2"
    # 2) Your bbox in WGS84 (lon/lat):
    BBOX = (46.150506,39.029564,46.196774,39.065496)  # (min_lon, min_lat, max_lon, max_lat)
    # 3) Output
    OUT = "crop_rgb.tif"

    crop_tci_to_bbox(TCI, BBOX, OUT)
    tif_to_png(OUT, "crop_rgb.png")
    print(f"Wrote {OUT}")

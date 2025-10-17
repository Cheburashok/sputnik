"""Utilities for extracting and converting satellite imagery from SAFE archives."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def extract_tci_from_safe(
    safe_zip_path: Path,
    output_path: Path,
    *,
    max_size_mb: float = 10.0,
    quality: int = 100,
) -> Path:
    """Extract TCI (True Color Image) from SAFE archive and convert to PNG.

    Finds the TCI_10m.jp2 file in the SAFE archive, converts it to PNG,
    and optimizes the size for web usage.

    Parameters
    ----------
    safe_zip_path:
        Path to the .SAFE.zip archive downloaded from Copernicus.
    output_path:
        Path where the PNG should be saved.
    max_size_mb:
        Maximum output file size in MB. If exceeded, image is downscaled.
        Default is 10 MB.
    quality:
        PNG compression quality (0-100). Default is 85.

    Returns
    -------
    Path:
        Path to the generated PNG file.

    Raises
    ------
    FileNotFoundError:
        If TCI file is not found in the archive.
    ValueError:
        If the archive is invalid or corrupted.

    Examples
    --------
    >>> # Extract and convert TCI to PNG
    >>> png_path = extract_tci_from_safe(
    ...     Path("data/S2A_MSIL1C_...zip"),
    ...     Path("output/preview.png"),
    ...     max_size_mb=5.0
    ... )
    """
    if not safe_zip_path.exists():
        raise FileNotFoundError(f"SAFE archive not found: {safe_zip_path}")

    logger.info(f"Extracting TCI from {safe_zip_path.name}")

    # Find TCI file in archive
    tci_path = None
    with zipfile.ZipFile(safe_zip_path, 'r') as zf:
        for name in zf.namelist():
            # Look for TCI_10m.jp2 in GRANULE/.../IMG_DATA/R10m/
            if name.endswith("_TCI_10m.jp2") and "/R10m/" in name:
                tci_path = name
                logger.debug(f"Found TCI: {tci_path}")
                break

        if not tci_path:
            raise FileNotFoundError(
                f"TCI_10m.jp2 not found in {safe_zip_path.name}. "
                "This might not be a Sentinel-2 L1C product."
            )

        # Extract and open TCI
        logger.info("Reading TCI from archive...")
        with zf.open(tci_path) as tci_file:
            img = Image.open(tci_file)
            img.load()  # Load image data before file closes

    # Convert to RGB if needed
    if img.mode != 'RGB':
        logger.debug(f"Converting from {img.mode} to RGB")
        img = img.convert('RGB')

    # Optimize size
    original_size = img.size
    logger.info(f"Original size: {img.size[0]}x{img.size[1]}")

    # Estimate file size and downscale if needed
    max_size_bytes = max_size_mb * 1024 * 1024
    img = _optimize_image_size(img, max_size_bytes, quality)

    # Save as PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, 'PNG', optimize=True, quality=quality)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Saved PNG: {output_path.name} "
        f"({img.size[0]}x{img.size[1]}, {file_size_mb:.2f} MB)"
    )

    return output_path


def _optimize_image_size(img: Image.Image, max_bytes: float, quality: int) -> Image.Image:
    """Downscale image if estimated size exceeds maximum.

    Uses rough estimation: RGB image ≈ width × height × 3 bytes (uncompressed).
    PNG compression typically achieves 30-50% of uncompressed size.
    """
    # Estimate uncompressed size
    width, height = img.size
    estimated_uncompressed = width * height * 3

    # Assume PNG achieves ~40% compression
    estimated_compressed = estimated_uncompressed * 0.4

    if estimated_compressed <= max_bytes:
        return img

    # Calculate required scale factor
    scale_factor = (max_bytes / estimated_compressed) ** 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    logger.info(
        f"Downscaling to {new_width}x{new_height} "
        f"(scale: {scale_factor:.2f}) to fit size limit"
    )

    # Use LANCZOS for high-quality downscaling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def batch_extract_tcis(
    safe_archives: list[Path],
    output_dir: Path,
    *,
    max_size_mb: float = 10.0,
    quality: int = 85,
) -> list[Path]:
    """Extract TCI from multiple SAFE archives.

    Parameters
    ----------
    safe_archives:
        List of paths to SAFE zip archives.
    output_dir:
        Directory where PNGs will be saved.
    max_size_mb:
        Maximum size for each PNG in MB.
    quality:
        PNG quality (0-100).

    Returns
    -------
    list[Path]:
        List of successfully created PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_files: list[Path] = []

    for i, safe_path in enumerate(safe_archives, 1):
        try:
            # Generate output filename from SAFE name
            png_name = safe_path.stem.replace('.SAFE', '') + '.png'
            output_path = output_dir / png_name

            logger.info(f"Processing {i}/{len(safe_archives)}: {safe_path.name}")

            png_path = extract_tci_from_safe(
                safe_path,
                output_path,
                max_size_mb=max_size_mb,
                quality=quality
            )
            png_files.append(png_path)

        except Exception as e:
            logger.error(f"Failed to process {safe_path.name}: {e}")
            continue

    logger.info(f"Successfully created {len(png_files)}/{len(safe_archives)} PNGs")
    return png_files


__all__ = [
    "extract_tci_from_safe",
    "batch_extract_tcis",
]

#!/usr/bin/env python3
"""
Example usage of the new multi-band extraction feature in MonumentImageManager.

This demonstrates how to use the bands parameter to extract specific Sentinel-2
bands and merge them into a GeoTIFF file.
"""

from pathlib import Path
from copernicus_utils.manager import MonumentImageManager
import datetime as dt

# Example 1: Using default 6 bands [B02, B03, B04, B08, B11, B12]
# This will extract Blue, Green, Red, NIR, SWIR1, SWIR2 bands
def summer_default_bands():
    """Extract default 6 bands for all monuments."""
    manager = MonumentImageManager(
        monuments_csv=Path("data/monuments.csv"),
        storage_root=Path("data/collections"),
        max_cloud_cover=30.0,
        asset_type="PRODUCT",
    )

    # The bands parameter defaults to ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    for year in range(2015, 2025):
        results = manager.collect_time_series_for_all_monuments(
            start_date=dt.date(year, 6, 1),
            end_date=dt.date(year, 9, 20),
        )

    print(f"Downloaded {sum(len(paths) for paths in results.values())} images")
    return results


# Example 2: Using custom bands (only RGB + NIR at 10m resolution)
def example_custom_bands():
    """Extract only RGB and NIR bands."""
    manager = MonumentImageManager(
        monuments_csv=Path("data/monuments.csv"),
        storage_root=Path("data/collections"),
        max_cloud_cover=30.0,
        asset_type="PRODUCT",
    )

    # Custom bands: Blue, Green, Red, NIR (all at 10m resolution)
    results = manager.collect_time_series_for_all_monuments(
        start_date=dt.date(2023, 1, 1),
        end_date=dt.date(2023, 12, 31),
        interval=30,
        bands=['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
    )

    print(f"Downloaded {sum(len(paths) for paths in results.values())} images")
    return results


# Example 3: Using only SWIR bands for geological analysis
def example_swir_bands():
    """Extract only SWIR bands for geological/mineral analysis."""
    manager = MonumentImageManager(
        monuments_csv=Path("data/monuments.csv"),
        storage_root=Path("data/collections"),
        max_cloud_cover=30.0,
        asset_type="PRODUCT",
    )

    # SWIR bands at 20m resolution
    results = manager.collect_time_series_for_all_monuments(
        start_date=dt.date(2023, 1, 1),
        end_date=dt.date(2023, 12, 31),
        interval=30,
        bands=['B11', 'B12']  # SWIR1, SWIR2
    )

    print(f"Downloaded {sum(len(paths) for paths in results.values())} images")
    return results


# Example 4: Reading the resulting multi-band GeoTIFF
def example_read_multiband_tif():
    """Demonstrate how to read the resulting multi-band GeoTIFF."""
    import rasterio
    import numpy as np

    # Path to a downloaded multi-band GeoTIFF
    tif_path = Path("data/collections/Monument1/images/example.tif")

    if tif_path.exists():
        with rasterio.open(tif_path) as src:
            # Get band count
            num_bands = src.count
            print(f"Number of bands: {num_bands}")

            # Read all bands
            data = src.read()  # Shape: (bands, height, width)
            print(f"Data shape: {data.shape}")

            # Read individual bands
            blue = src.read(1)   # B02 (Blue)
            green = src.read(2)  # B03 (Green)
            red = src.read(3)    # B04 (Red)
            nir = src.read(4)    # B08 (NIR)
            swir1 = src.read(5)  # B11 (SWIR1)
            swir2 = src.read(6)  # B12 (SWIR2)

            # Calculate NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir.astype(float) - red.astype(float)) / (nir + red + 1e-10)
            print(f"NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")

            # Get georeferencing info
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
    else:
        print(f"File not found: {tif_path}")


# Example 5: Download full SAFE archives without processing
def example_download_full_safe_archives():
    """Download complete SAFE archives to data/archives/ for later processing.

    This is useful when you want:
    - Access to ALL bands (not just the 6 default ones)
    - Access to all metadata and auxiliary data
    - To process the data later with custom tools
    - To keep the original data for archival purposes
    """
    manager = MonumentImageManager(
        monuments_csv=Path("data/monuments.csv"),
        storage_root=Path("data/collections"),
        max_cloud_cover=30.0,
        asset_type="PRODUCT",
    )

    # Download full SAFE archives to data/archives/
    # Note: Each SAFE archive is ~800MB-1GB, so this will use significant disk space
    results = manager.download_safe_archives_for_all_monuments(
        start_date=dt.date(2025, 6, 1),
        end_date=dt.date(2025, 7, 31),
        archives_dir=Path("data/archives")  # Custom directory for archives
    )

    print(f"Downloaded {sum(len(paths) for paths in results.values())} SAFE archives")

    # The archives are stored in:
    # data/archives/{monument_name}/{timestamp}.zip
    # data/archives/{monument_name}/metadata/{timestamp}.json

    return results


# Example 6: Processing downloaded SAFE archives
def example_process_safe_archive():
    """Example of manually processing a downloaded SAFE archive.

    Once you have downloaded SAFE archives, you can process them
    in various ways depending on your needs.
    """
    import zipfile

    # Path to a downloaded SAFE archive
    archive_path = Path("data/archives/Monument1/2023-06-15T07-46-19_024000Z__46_153049_39_037053_46_207037_39_061982.zip")

    if archive_path.exists():
        print(f"Processing SAFE archive: {archive_path.name}")

        with zipfile.ZipFile(archive_path, 'r') as zf:
            # List all files in the archive
            all_files = zf.namelist()
            print(f"Total files in archive: {len(all_files)}")

            # Find all band files
            band_files = [f for f in all_files if f.endswith('.jp2') and '/IMG_DATA/' in f]
            print(f"\nAvailable bands ({len(band_files)}):")
            for band_file in sorted(band_files):
                # Extract band name from file path
                band_name = Path(band_file).stem.split('_')[-1]
                print(f"  - {band_name}: {band_file}")

            # Find metadata files
            metadata_files = [f for f in all_files if f.endswith('.xml')]
            print(f"\nMetadata files: {len(metadata_files)}")

            # Example: Extract a specific band
            # Find B05 (Red Edge 1) at 20m resolution
            b05_files = [f for f in band_files if 'B05_20m' in f]
            if b05_files:
                print(f"\nExtracting B05 (Red Edge 1): {b05_files[0]}")
                # You can extract and process this band
                # with zf.open(b05_files[0]) as band_file:
                #     # Process the band data
                #     pass
    else:
        print(f"Archive not found: {archive_path}")
        print("Run example_download_full_safe_archives() first to download archives")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("Multi-band Extraction Examples")
    print("=" * 80)

    print("\nExample 1: Default 6 bands (B02, B03, B04, B08, B11, B12)")
    print("This extracts Blue, Green, Red, NIR, SWIR1, SWIR2")
    print("The resulting GeoTIFF will have 6 bands in this order.")
    summer_default_bands()

    # print("\nExample 2: Custom RGB + NIR bands")
    # print("You can specify any combination of bands you need.")

    # print("\nExample 3: SWIR bands only")
    # print("Useful for geological or mineral analysis.")

    # print("\nExample 4: Reading multi-band GeoTIFF")
    # print("Shows how to work with the resulting files.")

    # print("\nExample 5: Download full SAFE archives")
    # print("Downloads complete SAFE zip files to data/archives/")
    # print("Useful when you need access to ALL bands and metadata")
    # print("Note: Each archive is ~800MB-1GB")
    # # example_download_full_safe_archives()

    # print("\nExample 6: Processing downloaded SAFE archives")
    # print("Shows how to manually work with downloaded archives")

    # print("\n" + "=" * 80)
    # print("Available Sentinel-2 Bands:")
    # print("=" * 80)
    # print("10m resolution: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)")
    # print("20m resolution: B05, B06, B07, B8A, B11 (SWIR1), B12 (SWIR2)")
    # print("60m resolution: B01, B09, B10")
    # print("\nNote: When mixing resolutions, lower resolution bands are")
    # print("automatically upsampled to match the highest resolution using")
    # print("bilinear interpolation.")
    # print("\n" + "=" * 80)
    # print("Usage Modes:")
    # print("=" * 80)
    # print("1. Extract specific bands as multi-band GeoTIFF (Examples 1-3)")
    # print("   - Fast processing, only extracts what you need")
    # print("   - Results in smaller files")
    # print("   - Good for most analysis workflows")
    # print("\n2. Download full SAFE archives (Example 5)")
    # print("   - No processing, just downloads the raw data")
    # print("   - Access to all bands and metadata")
    # print("   - Larger files, but maximum flexibility")
    # print("=" * 80)

"""High level orchestration utilities for populating a local cache."""

from __future__ import annotations

import csv
import datetime as dt
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

from .collection import GeospatialCollection
from .copernicus_api import (
    AssetType,
    BBox,
    Collection,
    CopernicusAPIError,
    CopernicusImage,
    _download_asset_to_file,
    _read_token,
    fetch_all_images_in_period,
    fetch_latest_image_for_bbox,
)
from .image_processing import extract_tci_from_safe
from .jp_processing import crop_tci_to_bbox, tif_to_png


@dataclass(slots=True)
class Monument:
    name: str
    bbox: BBox


class MonumentImageManager:
    """Download and cache Copernicus imagery for a list of monuments."""

    def __init__(
        self,
        monuments_csv: Path | str,
        *,
        storage_root: Path | str = Path("data/collections"),
        lookback_days: int = 30,
        max_cloud_cover: float = 30.0,
        collection: Collection = "SENTINEL-2",
        asset_type: AssetType = "PRODUCT",
        token_path: Optional[Path] = None,
    ) -> None:
        self.monuments_csv = Path(monuments_csv)
        self.storage_root = Path(storage_root)
        self.lookback_days = lookback_days
        self.max_cloud_cover = max_cloud_cover
        self.collection = collection
        self.asset_type = asset_type
        self.token_path = token_path
        self.monuments = self._load_monuments(self.monuments_csv)
        self._search_order = "asc"  # Default search order

    def _load_monuments(self, csv_path: Path) -> List[Monument]:
        monuments: List[Monument] = []
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                try:
                    bbox = (
                        float(row["min_lon"]),
                        float(row["min_lat"]),
                        float(row["max_lon"]),
                        float(row["max_lat"]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed row in monuments CSV: {row}"  # pragma: no cover
                    ) from exc
                monuments.append(Monument(name=row["name"], bbox=bbox))
        return monuments

    # ------------------------------------------------------------------
    # Helper methods
    def _get_storage_directories(self, collection: GeospatialCollection) -> tuple[Path, Path]:
        """Get image and metadata directories based on asset type."""
        if self.asset_type == "QUICKLOOK":
            images_dir = collection.collection_dir / "quicklook"
            metadata_dir = collection.collection_dir / "quicklook_metadata"
        else:  # PRODUCT
            images_dir = collection.images_dir
            metadata_dir = collection.metadata_dir
        return images_dir, metadata_dir

    def _parse_image_timestamp(self, metadata: dict) -> Optional[dt.datetime]:
        """Parse ISO8601 timestamp from image metadata."""
        timestamp_str = metadata.get("timestamp")
        if timestamp_str:
            return dt.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return None

    def _should_skip_existing(self, metadata_path: Path, item: dict) -> bool:
        """Check if image already exists and should be skipped."""
        if metadata_path.exists():
            logger.info(f"  Skipping {item['index']}/{item['total']}: already exists")
            return True
        return False

    def _should_skip_by_interval(
        self,
        image_date: Optional[dt.datetime],
        last_downloaded_date: Optional[dt.datetime],
        interval: Optional[int],
        item: dict
    ) -> bool:
        """Check if image should be skipped based on interval filtering."""
        if interval is None or last_downloaded_date is None or image_date is None:
            return False

        days_since_last = (image_date - last_downloaded_date).days
        if days_since_last < interval:
            logger.info(
                f"  Skipping {item['index']}/{item['total']}: "
                f"only {days_since_last} days since last image (interval={interval})"
            )
            return True
        return False

    def _save_metadata(self, metadata_path: Path, metadata: dict) -> None:
        """Save image metadata to JSON file."""
        import json
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8"
        )

    def _extract_tci_from_safe_archive(self, zip_path: Path, monument: Monument, base: str, images_dir: Path) -> Path:
        """Extract and process TCI from SAFE archive.

        Downloads SAFE archive, extracts TCI, crops to bbox, and converts to PNG.
        """
        # Find and extract TCI_10m.jp2 from SAFE archive
        with zipfile.ZipFile(zip_path, 'r') as zf:
            tci_jp2_path = self._find_tci_in_archive(zf)

            # Extract TCI to temporary location
            with tempfile.TemporaryDirectory() as extract_dir:
                extract_path = Path(extract_dir)
                tci_extracted = extract_path / Path(tci_jp2_path).name

                # Extract the TCI file
                with zf.open(tci_jp2_path) as source:
                    tci_extracted.write_bytes(source.read())

                # Crop TCI to monument bbox
                cropped_tif = extract_path / "cropped.tif"
                crop_tci_to_bbox(
                    str(tci_extracted),
                    monument.bbox,
                    str(cropped_tif),
                    overviews=False  # Don't need overviews for PNG conversion
                )

                # Convert cropped TIF to PNG
                png_path = images_dir / f"{base}.png"
                tif_to_png(
                    str(cropped_tif),
                    str(png_path),
                    normalize=True
                )

                return png_path

    def _extract_bands_from_safe_archive(
        self,
        zip_path: Path,
        monument: Monument,
        base: str,
        images_dir: Path,
        bands: List[str]
    ) -> Path:
        """Extract and merge specified bands from SAFE archive.

        Downloads SAFE archive, extracts specified bands, merges them,
        crops to bbox, and saves as GeoTIFF.

        Parameters
        ----------
        zip_path:
            Path to the SAFE archive zip file.
        monument:
            Monument with bbox information.
        base:
            Base filename for output.
        images_dir:
            Directory where output files should be saved.
        bands:
            List of band names to extract (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).

        Returns
        -------
        Path:
            Path to the generated GeoTIFF file.

        Notes
        -----
        Band files in SAFE archives are named like T38SPJ_20250601T073619_B02.jp2
        and are located directly in the IMG_DATA directory. Bands at different
        resolutions (10m, 20m, 60m) are automatically resampled to match the
        resolution of the first band in the list using bilinear interpolation.
        """
        import numpy as np
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds

        with zipfile.ZipFile(zip_path, 'r') as zf:
            with tempfile.TemporaryDirectory() as extract_dir:
                extract_path = Path(extract_dir)

                # Find and extract each band
                band_files = []
                for band in bands:
                    band_jp2_path = self._find_band_in_archive(zf, band)
                    band_extracted = extract_path / Path(band_jp2_path).name

                    # Extract the band file
                    with zf.open(band_jp2_path) as source:
                        band_extracted.write_bytes(source.read())

                    band_files.append(band_extracted)

                # Read and merge bands
                # Use the first band as reference for georeferencing and target resolution
                with rasterio.open(band_files[0]) as ref_src:
                    # Reproject bbox to image CRS
                    min_lon, min_lat, max_lon, max_lat = monument.bbox
                    bbox_dst = transform_bounds(
                        "EPSG:4326", ref_src.crs,
                        min_lon, min_lat, max_lon, max_lat,
                        densify_pts=21
                    )

                    # Store bbox bounds for reuse
                    ref_bounds = bbox_dst

                    # Read each band and resample to reference resolution
                    band_arrays = []
                    output_transform = None
                    output_shape = None

                    for i, band_file in enumerate(band_files):
                        with rasterio.open(band_file) as src:
                            # For the first band (reference), compute the output window
                            if i == 0:
                                # Compute pixel window intersecting the bbox
                                win = from_bounds(*ref_bounds, transform=src.transform)
                                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                                if win.width <= 0 or win.height <= 0:
                                    raise ValueError("BBox does not intersect the image.")

                                # Read the reference band
                                data = src.read(1, window=win, boundless=False)
                                output_shape = data.shape
                                output_transform = rasterio.windows.transform(win, src.transform)
                                band_arrays.append(data)

                            else:
                                # For other bands, compute window in their own resolution
                                win = from_bounds(*ref_bounds, transform=src.transform)
                                win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                                # Read the band
                                data = src.read(1, window=win, boundless=False)

                                # If band has different resolution, resample to reference resolution
                                if data.shape != output_shape:
                                    from scipy.ndimage import zoom
                                    zoom_factors = (
                                        output_shape[0] / data.shape[0],
                                        output_shape[1] / data.shape[1]
                                    )
                                    data = zoom(data, zoom_factors, order=1)  # Bilinear interpolation

                                band_arrays.append(data)

                    # Stack bands into a single multi-band array
                    merged_data = np.stack(band_arrays, axis=0)

                    # Use output dimensions from reference band
                    transform = output_transform
                    height = output_shape[0]
                    width = output_shape[1]

                    # Prepare output profile
                    profile = ref_src.profile.copy()
                    profile.update(
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=len(bands),
                        transform=transform,
                        compress="DEFLATE",
                        tiled=True,
                        BIGTIFF="IF_SAFER",
                    )

                    # Write merged GeoTIFF
                    tif_path = images_dir / f"{base}.tif"
                    with rasterio.open(tif_path, "w", **profile) as dst:
                        dst.write(merged_data)

                    return tif_path

    def _find_tci_in_archive(self, zf: zipfile.ZipFile) -> str:
        """Find TCI file path in SAFE archive."""
        for name in zf.namelist():
            # Look for TCI file (10m resolution)
            if "_TCI_10m.jp2" in name or "_TCI.jp2" in name:
                return name
        raise FileNotFoundError("TCI file not found in SAFE archive")

    def _find_band_in_archive(self, zf: zipfile.ZipFile, band: str) -> str:
        """Find a specific band file path in SAFE archive.

        Parameters
        ----------
        zf:
            Open ZipFile object for the SAFE archive.
        band:
            Band name to find (e.g., 'B02', 'B03', 'B04', 'B08', 'B11', 'B12').

        Returns
        -------
        str:
            Path to the band file within the archive.

        Raises
        ------
        FileNotFoundError:
            If the band file is not found in the archive.

        Notes
        -----
        Sentinel-2 band files are typically named like:
        - T38SPJ_20250601T073619_B02.jp2 (for band B02)
        - T38SPJ_20250601T073619_B11.jp2 (for band B11)

        Band resolutions:
        - 10m resolution: B02, B03, B04, B08
        - 20m resolution: B05, B06, B07, B8A, B11, B12
        - 60m resolution: B01, B09, B10
        """
        # Look for the band file in IMG_DATA directory
        # Pattern: .../IMG_DATA/..._{band}.jp2
        for name in zf.namelist():
            # Check if this is in IMG_DATA directory and matches the band
            if "/IMG_DATA/" in name and name.endswith(f"_{band}.jp2"):
                return name

        # If not found with underscore, try without (some archives may vary)
        for name in zf.namelist():
            if "/IMG_DATA/" in name and f"_{band}." in name and name.endswith(".jp2"):
                return name

        raise FileNotFoundError(
            f"Band {band} not found in SAFE archive IMG_DATA directory"
        )

    def _download_quicklook_asset(self, href: str, token: str, image_path: Path) -> None:
        """Download QUICKLOOK asset (small file, to memory)."""
        from .copernicus_api import _download_asset
        content = _download_asset(href, token)
        image_path.write_bytes(content)

    def _download_product_asset(
        self,
        href: str,
        token: str,
        monument: Monument,
        base: str,
        images_dir: Path,
        bands: Optional[List[str]] = None
    ) -> Path:
        """Download and process PRODUCT asset (large file, stream to disk).

        Downloads SAFE archive, extracts either TCI or specified bands,
        crops to bbox, and saves as PNG or GeoTIFF.

        Parameters
        ----------
        href:
            URL to download the asset from.
        token:
            Access token for authentication.
        monument:
            Monument with bbox information.
        base:
            Base filename for output.
        images_dir:
            Directory where output files should be saved.
        bands:
            Optional list of band names to extract (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
            If None, extracts TCI and saves as PNG.

        Returns
        -------
        Path:
            Path to the generated PNG or GeoTIFF file.
        """
        # Download to temp file, extract bands or TCI
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            # Download SAFE archive to temp file
            _download_asset_to_file(href, token, tmp_path)

            # Extract and process based on bands parameter
            if bands is not None:
                # Extract specified bands and save as GeoTIFF
                return self._extract_bands_from_safe_archive(
                    tmp_path, monument, base, images_dir, bands
                )
            else:
                # Extract TCI and save as PNG (legacy behavior)
                return self._extract_tci_from_safe_archive(
                    tmp_path, monument, base, images_dir
                )
        finally:
            # Clean up temporary SAFE archive
            if tmp_path.exists():
                tmp_path.unlink()

    # ------------------------------------------------------------------
    # Fetching
    def fetch_latest_for_monument(self, monument: Monument) -> Optional[CopernicusImage]:
        try:
            return fetch_latest_image_for_bbox(
                monument.bbox,
                lookback_days=self.lookback_days,
                max_cloud_cover=self.max_cloud_cover,
                collection=self.collection,
                token_path=self.token_path,
            )
        except CopernicusAPIError as e:
            logger.warning(f"Failed to fetch image for monument '{monument.name}': {e}")
            return None

    def collect_latest_images(self) -> List[Path]:
        """Fetch the latest image for each monument and persist it to disk."""

        stored_paths: List[Path] = []
        for monument in self.monuments:
            image = self.fetch_latest_for_monument(monument)
            if image is None:
                continue

            collection = GeospatialCollection(self.storage_root, monument.name)
            stored = collection.save(image)
            stored_paths.append(stored.image_path)
        return stored_paths

    def _process_single_image(
        self,
        item: dict,
        monument: Monument,
        collection: GeospatialCollection,
        images_dir: Path,
        metadata_dir: Path,
        last_downloaded_date: Optional[dt.datetime],
        interval: Optional[int],
        bands: Optional[List[str]] = None
    ) -> tuple[Optional[Path], Optional[dt.datetime]]:
        """Process a single image from the API response.

        Parameters
        ----------
        bands:
            Optional list of band names to extract (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
            If None, extracts TCI and saves as PNG.

        Returns
        -------
        tuple[Optional[Path], Optional[dt.datetime]]:
            Tuple of (image_path, updated_last_download_date).
            Returns (None, last_date) if image was skipped.
        """
        # Build output path based on metadata
        base = collection._build_base_filename(item["metadata"])
        image_path = images_dir / f"{base}{item['extension']}"
        metadata_path = metadata_dir / f"{base}.json"

        # Parse image timestamp
        image_date = self._parse_image_timestamp(item["metadata"])

        # Skip if already downloaded
        if self._should_skip_existing(metadata_path, item):
            # Update last downloaded date if we have this image
            updated_last_date = image_date if image_date else last_downloaded_date
            return image_path, updated_last_date

        # Apply interval filter if specified
        if self._should_skip_by_interval(image_date, last_downloaded_date, interval, item):
            return None, last_downloaded_date

        # Download and process the image
        token = _read_token(item["token_path"])
        logger.info(
            f"  Downloading {item['index']}/{item['total']}: "
            f"{item['metadata']['timestamp']}"
        )

        # Download based on asset type
        if self.asset_type == "QUICKLOOK":
            self._download_quicklook_asset(item["href"], token, image_path)
        else:
            image_path = self._download_product_asset(
                item["href"], token, monument, base, images_dir, bands
            )

        # Save metadata
        self._save_metadata(metadata_path, item["metadata"])

        return image_path, image_date

    def collect_time_series_for_monument(
        self,
        monument: Monument,
        start_date: dt.date | dt.datetime,
        end_date: dt.date | dt.datetime,
        interval: Optional[int] = None,
        bands: Optional[List[str]] = None
    ) -> List[Path]:
        """Fetch all high-quality images for a monument within a date range.

        Downloads are streamed directly to disk to avoid loading large files
        (800MB+ each) into memory.

        The asset type (PRODUCT or QUICKLOOK) is determined by the manager's
        asset_type setting. Files are stored in different directories:
        - PRODUCT: collections/{monument}/images/
        - QUICKLOOK: collections/{monument}/quicklook/

        Parameters
        ----------
        monument:
            Monument to fetch images for.
        start_date:
            Start date of the period (inclusive).
        end_date:
            End date of the period (inclusive).
        interval:
            Minimum number of days between downloaded images. If None, all images
            are downloaded. If specified, only images that are at least 'interval'
            days apart from the previous downloaded image will be included.
        bands:
            Optional list of band names to extract (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
            If None, extracts TCI and saves as PNG. Only applicable when asset_type is PRODUCT.

        Returns
        -------
        List[Path]:
            List of paths to stored images, sorted chronologically.
        """
        logger.info(
            f"Collecting {self.asset_type} time series for '{monument.name}' "
            f"from {start_date} to {end_date}"
        )

        collection = GeospatialCollection(self.storage_root, monument.name)

        # Setup directories
        images_dir, metadata_dir = self._get_storage_directories(collection)
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        stored_paths: List[Path] = []
        last_downloaded_date: Optional[dt.datetime] = None

        try:
            # Use generator to stream downloads one at a time
            search_order = getattr(self, '_search_order', 'asc')
            image_generator = fetch_all_images_in_period(
                monument.bbox,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=self.max_cloud_cover,
                collection=self.collection,
                asset_type=self.asset_type,
                token_path=self.token_path,
                order=search_order,
            )

            for item in image_generator:
                try:
                    image_path, last_downloaded_date = self._process_single_image(
                        item, monument, collection, images_dir, metadata_dir,
                        last_downloaded_date, interval, bands
                    )
                    if image_path:
                        stored_paths.append(image_path)

                except Exception as e:
                    logger.warning(
                        f"  Failed to download image {item['index']}: {e}"
                    )
                    continue

            logger.info(
                f"Successfully stored {len(stored_paths)} images for '{monument.name}'"
            )
            return stored_paths

        except CopernicusAPIError as e:
            logger.warning(f"Failed to fetch time series for '{monument.name}': {e}")
            return []

    def collect_time_series_for_all_monuments(
        self,
        start_date: dt.date | dt.datetime,
        end_date: dt.date | dt.datetime,
        interval: Optional[int] = None,
        bands: Optional[List[str]] = None
    ) -> dict[str, List[Path]]:
        """Fetch all high-quality images for all monuments within a date range.

        Parameters
        ----------
        start_date:
            Start date of the period (inclusive).
        end_date:
            End date of the period (inclusive).
        interval:
            Minimum number of days between downloaded images. If None, all images
            are downloaded. If specified, only images that are at least 'interval'
            days apart from the previous downloaded image will be included.
        bands:
            List of band names to extract in the specified order.
            Default is ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] corresponding to
            (Blue, Green, Red, NIR, SWIR1, SWIR2).
            Extracts specified bands and saves as GeoTIFF. Only applicable when asset_type is PRODUCT.

        Returns
        -------
        dict[str, List[Path]]:
            Dictionary mapping monument names to lists of stored image paths.

        Notes
        -----
        Available Sentinel-2 bands:
        - B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) at 10m resolution
        - B11 (SWIR1), B12 (SWIR2) at 20m resolution
        When mixing resolutions, lower resolution bands are upsampled to match the highest resolution.
        """
        # Set default bands if not specified
        if bands is None:
            bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

        results: dict[str, List[Path]] = {}

        for monument in self.monuments:
            paths = self.collect_time_series_for_monument(
                monument, start_date, end_date, interval, bands
            )
            results[monument.name] = paths

        total_images = sum(len(paths) for paths in results.values())
        logger.info(
            f"Collected {total_images} total images across {len(self.monuments)} monuments"
        )

        return results

    def download_safe_archives_for_monument(
        self,
        monument: Monument,
        start_date: dt.date | dt.datetime,
        end_date: dt.date | dt.datetime,
        interval: Optional[int] = None,
        archives_dir: Optional[Path] = None
    ) -> List[Path]:
        """Download full SAFE archives for a monument within a date range.

        This downloads complete SAFE zip files without extracting or processing them.
        Useful when you need access to all bands and metadata, or want to process
        the data later with custom tools.

        Parameters
        ----------
        monument:
            Monument to fetch archives for.
        start_date:
            Start date of the period (inclusive).
        end_date:
            End date of the period (inclusive).
        interval:
            Minimum number of days between downloaded archives. If None, all archives
            are downloaded.
        archives_dir:
            Directory where SAFE archives should be stored. If None, uses
            storage_root/archives/{monument_name}/

        Returns
        -------
        List[Path]:
            List of paths to downloaded SAFE archive zip files.
        """
        logger.info(
            f"Downloading SAFE archives for '{monument.name}' "
            f"from {start_date} to {end_date}"
        )

        # Setup archives directory
        if archives_dir is None:
            archives_dir = self.storage_root / "archives" / monument.name
        archives_dir.mkdir(parents=True, exist_ok=True)

        # Also create metadata directory
        metadata_dir = archives_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        stored_paths: List[Path] = []
        last_downloaded_date: Optional[dt.datetime] = None

        try:
            # Use generator to stream downloads one at a time
            search_order = getattr(self, '_search_order', 'asc')
            image_generator = fetch_all_images_in_period(
                monument.bbox,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=self.max_cloud_cover,
                collection=self.collection,
                asset_type="PRODUCT",  # Always use PRODUCT for full archives
                token_path=self.token_path,
                order=search_order,
            )

            for item in image_generator:
                try:
                    # Build output path from metadata
                    from .collection import GeospatialCollection
                    collection = GeospatialCollection(self.storage_root, monument.name)
                    base = collection._build_base_filename(item["metadata"])
                    archive_path = archives_dir / f"{base}.zip"
                    metadata_path = metadata_dir / f"{base}.json"

                    # Parse image timestamp
                    image_date = self._parse_image_timestamp(item["metadata"])

                    # Skip if already downloaded
                    if metadata_path.exists():
                        logger.info(f"  Skipping {item['index']}/{item['total']}: already exists")
                        stored_paths.append(archive_path)
                        last_downloaded_date = image_date if image_date else last_downloaded_date
                        continue

                    # Apply interval filter if specified
                    if self._should_skip_by_interval(
                        image_date, last_downloaded_date, interval, item
                    ):
                        continue

                    # Download archive
                    token = _read_token(item["token_path"])
                    logger.info(
                        f"  Downloading {item['index']}/{item['total']}: "
                        f"{item['metadata']['timestamp']} ({archive_path.name})"
                    )

                    # Download directly to archive path (no processing)
                    _download_asset_to_file(item["href"], token, archive_path)

                    # Save metadata
                    self._save_metadata(metadata_path, item["metadata"])

                    stored_paths.append(archive_path)
                    last_downloaded_date = image_date

                except Exception as e:
                    logger.warning(
                        f"  Failed to download archive {item['index']}: {e}"
                    )
                    continue

            logger.info(
                f"Successfully downloaded {len(stored_paths)} SAFE archives for '{monument.name}'"
            )
            return stored_paths

        except CopernicusAPIError as e:
            logger.warning(f"Failed to fetch archives for '{monument.name}': {e}")
            return []

    def download_safe_archives_for_all_monuments(
        self,
        start_date: dt.date | dt.datetime,
        end_date: dt.date | dt.datetime,
        interval: Optional[int] = None,
        archives_dir: Optional[Path] = None
    ) -> dict[str, List[Path]]:
        """Download full SAFE archives for all monuments within a date range.

        Parameters
        ----------
        start_date:
            Start date of the period (inclusive).
        end_date:
            End date of the period (inclusive).
        interval:
            Minimum number of days between downloaded archives. If None, all archives
            are downloaded.
        archives_dir:
            Base directory where SAFE archives should be stored. If None, uses
            storage_root/archives/. Each monument gets its own subdirectory.

        Returns
        -------
        dict[str, List[Path]]:
            Dictionary mapping monument names to lists of archive paths.
        """
        results: dict[str, List[Path]] = {}

        for monument in self.monuments:
            monument_archives_dir = None
            if archives_dir is not None:
                monument_archives_dir = Path(archives_dir) / monument.name

            paths = self.download_safe_archives_for_monument(
                monument, start_date, end_date, interval, monument_archives_dir
            )
            results[monument.name] = paths

        total_archives = sum(len(paths) for paths in results.values())
        logger.info(
            f"Downloaded {total_archives} total SAFE archives across {len(self.monuments)} monuments"
        )

        return results

    def extract_previews_for_monument(
        self,
        monument: Monument,
        *,
        max_size_mb: float = 10.0,
        quality: int = 85,
    ) -> List[Path]:
        """Extract TCI preview PNGs from downloaded SAFE archives.

        Searches for .zip files in the monument's collection directory and
        extracts the TCI (True Color Image) to PNG format for web viewing.

        Parameters
        ----------
        monument:
            Monument to process.
        max_size_mb:
            Maximum PNG file size in MB. Default is 10 MB.
        quality:
            PNG quality (0-100). Default is 85.

        Returns
        -------
        List[Path]:
            List of paths to created PNG files.
        """
        collection = GeospatialCollection(self.storage_root, monument.name)
        previews_dir = collection.collection_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)

        # Find all .zip files (SAFE archives)
        zip_files = list(collection.images_dir.glob("*.zip"))

        if not zip_files:
            logger.warning(f"No SAFE archives found for '{monument.name}'")
            return []

        logger.info(
            f"Extracting previews for '{monument.name}' "
            f"({len(zip_files)} archives)"
        )

        preview_paths: List[Path] = []
        for i, zip_path in enumerate(zip_files, 1):
            try:
                # Generate PNG name from zip name
                png_name = zip_path.stem + ".png"
                output_path = previews_dir / png_name

                # Skip if already exists
                if output_path.exists():
                    logger.info(f"  {i}/{len(zip_files)}: Skipping {png_name} (exists)")
                    preview_paths.append(output_path)
                    continue

                logger.info(f"  {i}/{len(zip_files)}: Extracting {png_name}")

                png_path = extract_tci_from_safe(
                    zip_path,
                    output_path,
                    max_size_mb=max_size_mb,
                    quality=quality
                )
                preview_paths.append(png_path)

            except Exception as e:
                logger.warning(f"  Failed to extract preview from {zip_path.name}: {e}")
                continue

        logger.info(
            f"Created {len(preview_paths)} previews for '{monument.name}'"
        )
        return preview_paths

    def extract_previews_for_all_monuments(
        self,
        *,
        max_size_mb: float = 10.0,
        quality: int = 85,
    ) -> dict[str, List[Path]]:
        """Extract TCI preview PNGs for all monuments.

        Parameters
        ----------
        max_size_mb:
            Maximum PNG file size in MB. Default is 10 MB.
        quality:
            PNG quality (0-100). Default is 85.

        Returns
        -------
        dict[str, List[Path]]:
            Dictionary mapping monument names to lists of preview PNG paths.
        """
        results: dict[str, List[Path]] = {}

        for monument in self.monuments:
            paths = self.extract_previews_for_monument(
                monument,
                max_size_mb=max_size_mb,
                quality=quality
            )
            results[monument.name] = paths

        total_previews = sum(len(paths) for paths in results.values())
        logger.info(
            f"Created {total_previews} total previews across {len(self.monuments)} monuments"
        )

        return results


def _determine_search_order(start_dt: dt.datetime, end_dt: dt.datetime) -> tuple[str, dt.datetime, dt.datetime]:
    """Determine search order and normalize date range.

    Returns
    -------
    tuple[str, dt.datetime, dt.datetime]:
        Tuple of (order, normalized_start, normalized_end).
        If dates are reversed, swaps them and returns 'desc', otherwise 'asc'.
    """
    reverse_order = start_dt > end_dt
    if reverse_order:
        order = "desc"
        logger.info("Start date is after end date - using descending order")
        # Swap dates to create intervals, but we'll reverse the list
        return order, end_dt, start_dt
    else:
        order = "asc"
        logger.info("Using ascending (chronological) order")
        return order, start_dt, end_dt


def _create_date_intervals(
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    interval_days: int = 30,
    reverse: bool = False
) -> List[tuple[dt.datetime, dt.datetime]]:
    """Split date range into non-overlapping intervals.

    Parameters
    ----------
    start_dt:
        Start datetime (must be before or equal to end_dt).
    end_dt:
        End datetime.
    interval_days:
        Number of days per interval. Default is 30.
    reverse:
        If True, reverse the intervals list (newest first).

    Returns
    -------
    List[tuple[dt.datetime, dt.datetime]]:
        List of (interval_start, interval_end) tuples.
    """
    intervals = []
    current_start = start_dt

    while current_start < end_dt:
        # Use min() to handle the last interval which might be shorter
        current_end = min(
            current_start + dt.timedelta(days=interval_days - 1, hours=23, minutes=59, seconds=59),
            end_dt
        )
        intervals.append((current_start, current_end))
        # Move to next day to avoid overlap
        current_start = current_end + dt.timedelta(seconds=1)

    # If processing in reverse chronological order, reverse the intervals list
    if reverse:
        intervals.reverse()
        logger.info(
            f"Split date range into {len(intervals)} non-overlapping intervals of ~{interval_days} days each "
            f"(reversed for descending order - processing newest first)"
        )
    else:
        logger.info(
            f"Split date range into {len(intervals)} non-overlapping intervals of ~{interval_days} days each"
        )

    return intervals


def _merge_interval_results(
    merged_results: dict[str, List[Path]],
    interval_results: dict[str, List[Path]]
) -> None:
    """Merge results from a single interval into the accumulated results.

    Modifies merged_results in place.
    """
    for monument_name, paths in interval_results.items():
        if monument_name not in merged_results:
            merged_results[monument_name] = []
        merged_results[monument_name].extend(paths)


def demo_collect_all(
    start_date: str,
    end_date: str,
    monuments_csv: Path | str = Path("data/monuments.csv"),
    *,
    storage_root: Path | str = Path("data/collections"),
    max_cloud_cover: float = 100.0,
    token_path: Optional[Path] = None,
    collection: Collection = "SENTINEL-2",
) -> dict[str, List[Path]]:
    """Collect imagery for all monuments across a date range in 30-day intervals.

    The date range is split into non-overlapping 30-day intervals, processed sequentially
    to avoid file write conflicts.

    Parameters
    ----------
    start_date:
        Start date as ISO string (e.g., "2023-03-01").
    end_date:
        End date as ISO string (e.g., "2024-09-01").
    monuments_csv:
        Path to CSV file containing monument data (name, bbox coordinates).
    storage_root:
        Root directory for storing downloaded images.
    max_cloud_cover:
        Maximum acceptable cloud cover percentage (0-100). Default: 100%.
    token_path:
        Path to Copernicus access token file.
    collection:
        STAC collection name. Default: "SENTINEL-2".

    Returns
    -------
    dict[str, List[Path]]:
        Dictionary mapping monument names to lists of stored image paths.
    """
    # Parse date strings
    start_dt = dt.datetime.fromisoformat(start_date).replace(tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.fromisoformat(end_date).replace(tzinfo=dt.timezone.utc)

    # Determine search order and normalize date range
    order, normalized_start, normalized_end = _determine_search_order(start_dt, end_dt)
    reverse_order = (order == "desc")

    # Split date range into non-overlapping 30-day intervals
    intervals = _create_date_intervals(
        normalized_start,
        normalized_end,
        interval_days=30,
        reverse=reverse_order
    )

    # Create manager
    manager = MonumentImageManager(
        monuments_csv,
        storage_root=storage_root,
        lookback_days=30,  # Not used in time series collection
        max_cloud_cover=max_cloud_cover,
        token_path=token_path,
        collection=collection,
        asset_type="PRODUCT",
    )

    # Set search order
    manager._search_order = order

    # Process intervals sequentially to avoid file write conflicts
    merged_results: dict[str, List[Path]] = {}

    for i, (interval_start, interval_end) in enumerate(intervals, 1):
        logger.info(
            f"Processing interval {i}/{len(intervals)}: {interval_start.date()} to {interval_end.date()}"
        )

        interval_results = manager.collect_time_series_for_all_monuments(
            start_date=interval_start,
            end_date=interval_end,
            interval=None
        )

        # Merge results
        _merge_interval_results(merged_results, interval_results)

    # Log summary
    total_images = sum(len(paths) for paths in merged_results.values())
    logger.info(
        f"Completed collection: {total_images} total images across "
        f"{len(merged_results)} monuments"
    )

    return merged_results


__all__ = ["MonumentImageManager", "demo_collect_all", "Monument"]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Collect all")
    result = demo_collect_all(
        start_date="2023-12-02",
        end_date="1990-03-01",
    )
    total_images = sum(len(paths) for paths in result.values())
    print(f"Successfully collected {total_images} images across {len(result)} monuments")
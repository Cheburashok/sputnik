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
from multiprocessing import Pool

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

    def collect_time_series_for_monument(
        self,
        monument: Monument,
        start_date: dt.date | dt.datetime,
        end_date: dt.date | dt.datetime,
        interval: Optional[int] = None,
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

        # Use different directory based on asset type
        if self.asset_type == "QUICKLOOK":
            images_dir = collection.collection_dir / "quicklook"
            metadata_dir = collection.collection_dir / "quicklook_metadata"
        else:  # PRODUCT
            images_dir = collection.images_dir
            metadata_dir = collection.metadata_dir

        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        stored_paths: List[Path] = []
        last_downloaded_date: Optional[dt.datetime] = None

        try:
            # Use generator to stream downloads one at a time
            # Use the search order from manager (set by parallel processing)
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
                    # Build output path based on metadata
                    base = collection._build_base_filename(item["metadata"])
                    image_path = images_dir / f"{base}{item['extension']}"
                    metadata_path = metadata_dir / f"{base}.json"

                    # Parse image timestamp
                    timestamp_str = item["metadata"].get("timestamp")
                    if timestamp_str:
                        # Parse ISO8601 timestamp
                        image_date = dt.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    else:
                        image_date = None

                    # Skip if already downloaded
                    if metadata_path.exists():
                        logger.info(f"  Skipping {item['index']}/{item['total']}: already exists")
                        stored_paths.append(image_path)
                        # Update last downloaded date if we have this image
                        if image_date:
                            last_downloaded_date = image_date
                        continue

                    # Apply interval filter if specified
                    if interval is not None and last_downloaded_date is not None and image_date is not None:
                        days_since_last = (image_date - last_downloaded_date).days
                        if days_since_last < interval:
                            logger.info(
                                f"  Skipping {item['index']}/{item['total']}: "
                                f"only {days_since_last} days since last image (interval={interval})"
                            )
                            continue

                    # Get fresh token
                    token = _read_token(item["token_path"])
                    logger.info(
                        f"  Downloading {item['index']}/{item['total']}: "
                        f"{item['metadata']['timestamp']}"
                    )

                    # Use appropriate download method based on asset type
                    if self.asset_type == "QUICKLOOK":
                        # QUICKLOOK is small (~1MB), download to memory
                        from .copernicus_api import _download_asset
                        content = _download_asset(item["href"], token)
                        image_path.write_bytes(content)
                    else:
                        # PRODUCT is large (800MB+), stream to disk
                        # Download to temp file, extract TCI, crop, convert to PNG
                        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                            tmp_path = Path(tmp_file.name)

                        try:
                            # Download SAFE archive to temp file
                            _download_asset_to_file(item["href"], token, tmp_path)

                            # Find and extract TCI_10m.jp2 from SAFE archive
                            tci_jp2_path = None
                            with zipfile.ZipFile(tmp_path, 'r') as zf:
                                for name in zf.namelist():
                                    # Look for TCI file (10m resolution)
                                    if "_TCI_10m.jp2" in name or "_TCI.jp2" in name:
                                        tci_jp2_path = name
                                        break

                                if not tci_jp2_path:
                                    raise FileNotFoundError("TCI file not found in SAFE archive")

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

                                    # Update image_path to point to PNG
                                    image_path = png_path

                        finally:
                            # Clean up temporary SAFE archive
                            if tmp_path.exists():
                                tmp_path.unlink()

                    # Save metadata
                    import json
                    metadata_path.write_text(
                        json.dumps(item["metadata"], indent=2, sort_keys=True),
                        encoding="utf-8"
                    )

                    stored_paths.append(image_path)

                    # Update last downloaded date for interval tracking
                    if image_date:
                        last_downloaded_date = image_date

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

        Returns
        -------
        dict[str, List[Path]]:
            Dictionary mapping monument names to lists of stored image paths.
        """
        results: dict[str, List[Path]] = {}

        for monument in self.monuments:
            paths = self.collect_time_series_for_monument(
                monument, start_date, end_date, interval
            )
            results[monument.name] = paths

        total_images = sum(len(paths) for paths in results.values())
        logger.info(
            f"Collected {total_images} total images across {len(self.monuments)} monuments"
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


def _process_interval(args):
    """Helper function for parallel processing of date intervals.

    Parameters
    ----------
    args:
        Tuple containing (manager_params, interval_start, interval_end, order)

    Returns
    -------
    dict:
        Dictionary mapping monument names to lists of stored image paths.
    """
    manager_params, interval_start, interval_end, order = args

    # Recreate manager in subprocess
    manager = MonumentImageManager(**manager_params)

    # Store order in manager for passing to API
    manager._search_order = order

    logger.info(
        f"Processing interval: {interval_start.date()} to {interval_end.date()} "
        f"(order={order})"
    )

    return manager.collect_time_series_for_all_monuments(
        start_date=interval_start,
        end_date=interval_end,
        interval=None
    )


def demo_collect_all(
    start_date: str,
    end_date: str,
    monuments_csv: Path | str = Path("data/monuments.csv"),
    *,
    storage_root: Path | str = Path("data/collections"),
    max_cloud_cover: float = 100.0,
    token_path: Optional[Path] = None,
    collection: Collection = "SENTINEL-2",
    n_cpu: int = 4,
) -> dict[str, List[Path]]:
    """Collect imagery for all monuments across a date range using parallel processing.

    The date range is split into 30-day intervals, and each interval is processed
    in parallel using multiple CPUs.

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
    n_cpu:
        Number of CPUs to use for parallel processing. Default: 10.

    Returns
    -------
    dict[str, List[Path]]:
        Dictionary mapping monument names to lists of stored image paths.
    """
    # Parse date strings
    start_dt = dt.datetime.fromisoformat(start_date).replace(tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.fromisoformat(end_date).replace(tzinfo=dt.timezone.utc)

    # Determine search order based on date comparison
    reverse_order = start_dt > end_dt
    if reverse_order:
        order = "desc"
        logger.info("Start date is after end date - using descending order")
        # Swap dates to create intervals, but we'll reverse the list
        start_dt, end_dt = end_dt, start_dt
    else:
        order = "asc"
        logger.info("Using ascending (chronological) order")

    # Split date range into 30-day intervals
    intervals = []
    current_start = start_dt
    interval_days = 30

    while current_start < end_dt:
        current_end = min(current_start + dt.timedelta(days=interval_days), end_dt)
        intervals.append((current_start, current_end))
        current_start = current_end + dt.timedelta(seconds=1)  # Move to next interval

    # If processing in reverse chronological order, reverse the intervals list
    # so the most recent data is processed first
    if reverse_order:
        intervals.reverse()
        logger.info(
            f"Split date range into {len(intervals)} intervals of ~{interval_days} days each "
            f"(reversed for descending order - processing newest first)"
        )
    else:
        logger.info(
            f"Split date range into {len(intervals)} intervals of ~{interval_days} days each"
        )

    # Prepare manager parameters for subprocess recreation
    manager_params = {
        "monuments_csv": monuments_csv,
        "storage_root": storage_root,
        "lookback_days": 30,  # Not used in time series collection
        "max_cloud_cover": max_cloud_cover,
        "token_path": token_path,
        "collection": collection,
        "asset_type": "PRODUCT",
    }

    # Prepare arguments for parallel processing
    process_args = [
        (manager_params, interval_start, interval_end, order)
        for interval_start, interval_end in intervals
    ]

    # Process intervals in parallel
    logger.info(f"Starting parallel processing with {n_cpu} CPUs")

    with Pool(processes=n_cpu) as pool:
        results = pool.map(_process_interval, process_args)

    # Merge results from all intervals
    merged_results: dict[str, List[Path]] = {}
    for interval_result in results:
        for monument_name, paths in interval_result.items():
            if monument_name not in merged_results:
                merged_results[monument_name] = []
            merged_results[monument_name].extend(paths)

    # Log summary
    total_images = sum(len(paths) for paths in merged_results.values())
    logger.info(
        f"Completed parallel collection: {total_images} total images across "
        f"{len(merged_results)} monuments"
    )

    return merged_results


__all__ = ["MonumentImageManager", "demo_collect_all", "Monument"]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Collect all")
    result = demo_collect_all(
        start_date="2025-10-01",
        end_date="1990-03-01",
    )
    total_images = sum(len(paths) for paths in result.values())
    print(f"Successfully collected {total_images} images across {len(result)} monuments")
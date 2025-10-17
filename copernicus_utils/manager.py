"""High level orchestration utilities for populating a local cache."""

from __future__ import annotations

import csv
import datetime as dt
import logging
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

        try:
            # Use generator to stream downloads one at a time
            image_generator = fetch_all_images_in_period(
                monument.bbox,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=self.max_cloud_cover,
                collection=self.collection,
                asset_type=self.asset_type,
                token_path=self.token_path,
            )

            for item in image_generator:
                try:
                    # Build output path based on metadata
                    base = collection._build_base_filename(item["metadata"])
                    image_path = images_dir / f"{base}{item['extension']}"
                    metadata_path = metadata_dir / f"{base}.json"

                    # Skip if already downloaded
                    if metadata_path.exists():
                        logger.info(f"  Skipping {item['index']}/{item['total']}: already exists")
                        stored_paths.append(image_path)
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
                        _download_asset_to_file(item["href"], token, image_path)

                    # Save metadata
                    import json
                    metadata_path.write_text(
                        json.dumps(item["metadata"], indent=2, sort_keys=True),
                        encoding="utf-8"
                    )

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
    ) -> dict[str, List[Path]]:
        """Fetch all high-quality images for all monuments within a date range.

        Parameters
        ----------
        start_date:
            Start date of the period (inclusive).
        end_date:
            End date of the period (inclusive).

        Returns
        -------
        dict[str, List[Path]]:
            Dictionary mapping monument names to lists of stored image paths.
        """
        results: dict[str, List[Path]] = {}

        for monument in self.monuments:
            paths = self.collect_time_series_for_monument(
                monument, start_date, end_date
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


def demo_collect_all(
    monuments_csv: Path | str = Path("data/monuments.csv"),
    *,
    storage_root: Path | str = Path("data/collections"),
    lookback_days: int = 30,
    max_cloud_cover: float = 30.0,
    token_path: Optional[Path] = None,
    collection: Collection = "SENTINEL-2",
) -> List[Path]:
    """Example usage that collects the latest imagery for all monuments.

    Parameters
    ----------
    monuments_csv:
        Path to CSV file containing monument data (name, bbox coordinates).
    storage_root:
        Root directory for storing downloaded images.
    lookback_days:
        Number of days to look back from today. Default: 30 days (1 month).
    max_cloud_cover:
        Maximum acceptable cloud cover percentage (0-100). Default: 30%.
    token_path:
        Path to Copernicus access token file.
    collection:
        STAC collection name. Default: "SENTINEL-2".
    """

    manager = MonumentImageManager(
        monuments_csv,
        storage_root=storage_root,
        lookback_days=lookback_days,
        max_cloud_cover=max_cloud_cover,
        token_path=token_path,
        collection="SENTINEL-3",
        asset_type="QUICKLOOK"
    )
    return manager.collect_time_series_for_all_monuments(
        start_date=dt.date(2024, 6, 1),
        end_date=dt.date(2025, 10, 10),

    )


__all__ = ["MonumentImageManager", "demo_collect_all", "Monument"]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Collect all")
    result = demo_collect_all()
    print(f"Successfully collected {len(result)} images")
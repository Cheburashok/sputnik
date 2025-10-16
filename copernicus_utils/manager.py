"""High level orchestration utilities for populating a local cache."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .collection import GeospatialCollection
from .copernicus_api import (
    BBox,
    CopernicusAPIError,
    CopernicusImage,
    fetch_latest_image_for_bbox,
)


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
        delta_t: dt.timedelta = dt.timedelta(days=7),
        collection: str = "SENTINEL-2",
        token_path: Optional[Path] = None,
    ) -> None:
        self.monuments_csv = Path(monuments_csv)
        self.storage_root = Path(storage_root)
        self.delta_t = delta_t
        self.collection = collection
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
                delta=self.delta_t,
                collection=self.collection,
                token_path=self.token_path,
            )
        except CopernicusAPIError:
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


def demo_collect_all(
    monuments_csv: Path | str = Path("data/monuments.csv"),
    *,
    storage_root: Path | str = Path("data/collections"),
    delta_t: dt.timedelta = dt.timedelta(days=7),
    token_path: Optional[Path] = None,
    collection: str = "SENTINEL-2",
) -> List[Path]:
    """Example usage that collects the latest imagery for all monuments."""

    manager = MonumentImageManager(
        monuments_csv,
        storage_root=storage_root,
        delta_t=delta_t,
        token_path=token_path,
        collection=collection,
    )
    return manager.collect_latest_images()


__all__ = ["MonumentImageManager", "demo_collect_all", "Monument"]


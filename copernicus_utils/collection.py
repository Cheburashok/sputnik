"""Utilities for storing Copernicus images and metadata on disk."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

from .copernicus_api import CopernicusImage


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


@dataclass(slots=True)
class StoredImage:
    """Represents an image persisted on disk."""

    image_path: Path
    metadata_path: Path
    metadata: Mapping[str, object]


class GeospatialCollection:
    """Manage a directory of satellite imagery for a single monument."""

    def __init__(self, root: Path | str, name: str) -> None:
        self.root = Path(root)
        self.name = name
        self.collection_dir = self.root / _sanitize(name)
        self.images_dir = self.collection_dir / "images"
        self.metadata_dir = self.collection_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    def _build_base_filename(self, metadata: Mapping[str, object]) -> str:
        timestamp = str(metadata.get("timestamp", "unknown"))
        bbox = metadata.get("bbox")
        if isinstance(bbox, (list, tuple)):
            bbox_str = "_".join(f"{coord:.6f}" for coord in bbox)  # type: ignore[arg-type]
        else:
            bbox_str = str(bbox)
        safe_timestamp = _sanitize(timestamp.replace(":", "-"))
        return f"{safe_timestamp}__{_sanitize(bbox_str)}"

    def _entry_paths(self, metadata: Mapping[str, object], extension: str) -> Tuple[Path, Path]:
        base = self._build_base_filename(metadata)
        image_path = self.images_dir / f"{base}{extension}"
        metadata_path = self.metadata_dir / f"{base}.json"
        return image_path, metadata_path

    # ------------------------------------------------------------------
    # Public API
    def save(self, image: CopernicusImage, *, overwrite: bool = False) -> StoredImage:
        """Persist an image and its metadata to disk."""

        image_path, metadata_path = self._entry_paths(image.metadata, image.extension)

        if metadata_path.exists() and not overwrite:
            with metadata_path.open("r", encoding="utf-8") as fp:
                metadata = json.load(fp)
            existing_image = next(self.images_dir.glob(f"{metadata_path.stem}.*"), image_path)
            return StoredImage(image_path=existing_image, metadata_path=metadata_path, metadata=metadata)

        image_path.write_bytes(image.content)
        metadata_path.write_text(json.dumps(image.metadata, indent=2, sort_keys=True), encoding="utf-8")

        return StoredImage(image_path=image_path, metadata_path=metadata_path, metadata=image.metadata)

    def list_metadata(self) -> List[Mapping[str, object]]:
        """Return metadata dictionaries for all stored imagery."""

        items: List[Mapping[str, object]] = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as fp:
                items.append(json.load(fp))
        return items

    def iter_images(self) -> Iterable[StoredImage]:
        """Yield :class:`StoredImage` objects for every cached asset."""

        for metadata_file in sorted(self.metadata_dir.glob("*.json")):
            image_file = next(self.images_dir.glob(f"{metadata_file.stem}.*"), None)
            if image_file is None:
                continue
            with metadata_file.open("r", encoding="utf-8") as fp:
                metadata = json.load(fp)
            yield StoredImage(
                image_path=image_file,
                metadata_path=metadata_file,
                metadata=metadata,
            )

    def find(self, *, timestamp: str, bbox: Iterable[float]) -> Optional[StoredImage]:
        """Return the stored image for the given timestamp and bounding box."""

        metadata = {"timestamp": timestamp, "bbox": tuple(bbox)}
        image_path, metadata_path = self._entry_paths(metadata, "")
        # ``_entry_paths`` returns a filename with an empty extension.  We need to
        # match the actual extension stored on disk; search for files that share
        # the stem.
        stem = metadata_path.stem
        existing_metadata = next(self.metadata_dir.glob(f"{stem}.json"), None)
        existing_image = next(self.images_dir.glob(f"{stem}.*"), None)

        if existing_metadata and existing_image:
            with existing_metadata.open("r", encoding="utf-8") as fp:
                meta = json.load(fp)
            return StoredImage(image_path=existing_image, metadata_path=existing_metadata, metadata=meta)

        return None


__all__ = ["GeospatialCollection", "StoredImage"]


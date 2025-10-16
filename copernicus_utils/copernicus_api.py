"""Utilities for interacting with the Copernicus Dataspace API.

This module provides helper functions that read an access token from the
repository root, search for satellite products using the STAC endpoint and
download a renderable asset (thumbnail/preview) for the requested bounding box
and time frame.

The functions intentionally keep their surface area small so that they can be
reused by higher level orchestration code.  They raise ``RuntimeError`` when a
request cannot be fulfilled which allows callers to decide how to handle the
failure (retry, log, skip etc.).
"""

from __future__ import annotations

import datetime as dt
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import requests

TokenSource = Optional[Path]
BBox = Tuple[float, float, float, float]

STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"


class CopernicusAPIError(RuntimeError):
    """Raised when the Copernicus API cannot fulfill a request."""


@dataclass(slots=True)
class CopernicusImage:
    """Container for a downloaded Copernicus product asset."""

    content: bytes
    metadata: Mapping[str, Any]
    asset_href: str
    mime_type: str

    @property
    def extension(self) -> str:
        """File extension inferred from the ``mime_type``."""

        return mimetypes.guess_extension(self.mime_type) or ".bin"


def _read_token(token_path: TokenSource = None) -> str:
    """Read an OAuth token stored in ``.copernicus_token.txt``.

    Parameters
    ----------
    token_path:
        Optional path to a file containing the token.  When omitted the token
        is read from ``.copernicus_token.txt`` located in the repository root.
    """

    if token_path is None:
        token_path = Path(".copernicus_token.txt")
    elif not isinstance(token_path, Path):
        token_path = Path(token_path)

    if not token_path.exists():
        raise CopernicusAPIError(
            f"Token file not found at '{token_path}'. Please create the file "
            "with a valid Copernicus Dataspace access token."
        )

    return token_path.read_text(encoding="utf-8").strip()


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _build_datetime_range(date_or_range: Any) -> str:
    """Return an ISO8601 datetime range string accepted by STAC search."""

    if isinstance(date_or_range, str):
        return date_or_range

    if isinstance(date_or_range, dt.date) and not isinstance(date_or_range, dt.datetime):
        start = dt.datetime.combine(date_or_range, dt.time.min, tzinfo=dt.timezone.utc)
        end = start + dt.timedelta(days=1)
        return f"{start.isoformat()}/{end.isoformat()}"

    if isinstance(date_or_range, dt.datetime):
        start = date_or_range.astimezone(dt.timezone.utc)
        end = start + dt.timedelta(minutes=10)
        return f"{start.isoformat()}/{end.isoformat()}"

    if isinstance(date_or_range, Iterable):
        start, end = date_or_range  # type: ignore[misc]
        if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
            start = dt.datetime.combine(start, dt.time.min, tzinfo=dt.timezone.utc)
        if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
            end = dt.datetime.combine(end, dt.time.min, tzinfo=dt.timezone.utc)
        return f"{start.isoformat()}/{end.isoformat()}"

    raise TypeError(
        "date_or_range must be a datetime/date, an iterable of two datetimes, "
        "or an ISO8601 string."
    )


def _select_asset(feature: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    """Pick the most useful asset from a STAC feature."""

    assets: MutableMapping[str, Mapping[str, Any]] = feature.get("assets", {})  # type: ignore[assignment]
    preferred_keys = ("rendered_preview", "visual", "thumbnail", "quicklook")
    for key in preferred_keys:
        if key in assets:
            return key, assets[key]

    if assets:
        key = next(iter(assets.keys()))
        return key, assets[key]

    raise CopernicusAPIError("No downloadable asset found in the returned feature.")


def _search_features(
    bbox: BBox,
    datetime_range: Any,
    token: str,
    collection: str = "SENTINEL-2",
    limit: int = 1,
    order: str = "desc",
) -> Iterable[Mapping[str, Any]]:
    payload = {
        "bbox": list(bbox),
        "collections": [collection],
        "datetime": _build_datetime_range(datetime_range),
        "limit": limit,
        "sortby": [
            {
                "field": "properties.datetime",
                "direction": order,
            }
        ],
    }

    response = requests.post(STAC_SEARCH_URL, headers=_auth_headers(token), json=payload, timeout=60)
    if response.status_code != 200:
        raise CopernicusAPIError(
            "Copernicus STAC search failed with status "
            f"{response.status_code}: {response.text}"
        )

    body = response.json()
    features = body.get("features")
    if not features:
        raise CopernicusAPIError("No features returned for the given parameters.")

    return features


def _download_asset(href: str, token: str) -> bytes:
    response = requests.get(href, headers={"Authorization": f"Bearer {token}"}, timeout=120)
    if response.status_code != 200:
        raise CopernicusAPIError(
            f"Failed to download asset from {href}: {response.status_code} {response.text}"
        )
    return response.content


def fetch_image_for_bbox(
    bbox: BBox,
    date: Any,
    *,
    token_path: TokenSource = None,
    collection: str = "SENTINEL-2",
    limit: int = 1,
) -> CopernicusImage:
    """Fetch a Copernicus image for ``bbox`` on ``date``.

    Parameters
    ----------
    bbox:
        Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
    date:
        Either a date/datetime, an ISO8601 datetime range, or an iterable of
        two datetime objects describing the search window.
    token_path:
        Optional path to a file containing the access token.  Defaults to
        ``.copernicus_token.txt``.
    collection:
        STAC collection to query.  ``SENTINEL-2`` provides RGB imagery.
    limit:
        Number of products to request (the first one is returned).
    """

    token = _read_token(token_path)
    features = _search_features(bbox=bbox, datetime_range=date, token=token, collection=collection, limit=limit)
    feature = next(iter(features))
    asset_name, asset_info = _select_asset(feature)
    href = asset_info.get("href")
    if not href:
        raise CopernicusAPIError(
            f"Asset '{asset_name}' did not contain a download link (href missing)."
        )

    content = _download_asset(href, token)

    metadata = {
        "bbox": bbox,
        "timestamp": feature.get("properties", {}).get("datetime"),
        "cloud_cover": feature.get("properties", {}).get("eo:cloud_cover"),
        "collection": collection,
        "feature_id": feature.get("id"),
        "asset_name": asset_name,
        "asset_type": asset_info.get("type"),
    }

    mime_type = asset_info.get("type") or "application/octet-stream"

    return CopernicusImage(content=content, metadata=metadata, asset_href=href, mime_type=mime_type)


def fetch_latest_image_for_bbox(
    bbox: BBox,
    *,
    delta: dt.timedelta | None = dt.timedelta(days=7),
    token_path: TokenSource = None,
    collection: str = "SENTINEL-2",
) -> CopernicusImage:
    """Retrieve the most recent image for a bounding box.

    Parameters
    ----------
    delta:
        Look-back window.  Only items newer than ``now - delta`` are considered.
        When ``None`` all available imagery is considered.
    """

    if delta is None:
        datetime_range: str | Tuple[dt.datetime, dt.datetime] = ".."
    else:
        end = dt.datetime.now(dt.timezone.utc)
        start = end - delta
        datetime_range = (start, end)

    return fetch_image_for_bbox(
        bbox=bbox,
        date=datetime_range,
        token_path=token_path,
        collection=collection,
    )


def serialize_metadata(image: CopernicusImage) -> str:
    """Serialize metadata to JSON for persistence."""

    return json.dumps(image.metadata, indent=2, sort_keys=True)


__all__ = [
    "CopernicusImage",
    "CopernicusAPIError",
    "fetch_image_for_bbox",
    "fetch_latest_image_for_bbox",
    "serialize_metadata",
]


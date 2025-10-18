"""Utilities for interacting with the Copernicus Dataspace API.

This module provides helper functions that read an access token from the
repository root, search for satellite products using the STAC endpoint and
download full product data (typically .SAFE zip archives) for the requested
bounding box and time frame.

The functions intentionally keep their surface area small so that they can be
reused by higher level orchestration code.  They raise ``RuntimeError`` when a
request cannot be fulfilled which allows callers to decide how to handle the
failure (retry, log, skip etc.).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, MutableMapping, Optional, Tuple

import requests
import yaml

logger = logging.getLogger(__name__)

AssetType = Literal["PRODUCT", "QUICKLOOK"]
Collection = Literal["SENTINEL-2", "SENTINEL-3"]

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

        # Handle PRODUCT assets which are typically .SAFE zip archives
        if self.metadata.get("asset_name") == "PRODUCT":
            # Check if it's actually a zip by looking at magic bytes
            if self.content[:2] == b'PK':  # ZIP file magic number
                return ".zip"

        return mimetypes.guess_extension(self.mime_type) or ".bin"


def _generate_token_from_credentials(credentials_path: Path = Path(".credentials.yaml")) -> str:
    """Generate a fresh OAuth token from stored credentials.

    Parameters
    ----------
    credentials_path:
        Path to YAML file containing 'username' and 'password' fields.

    Returns
    -------
    str:
        Fresh access token valid for 10 minutes.
    """
    if not credentials_path.exists():
        raise CopernicusAPIError(
            f"Credentials file not found at '{credentials_path}'. "
            "Please create a .credentials.yaml file with username and password."
        )

    with credentials_path.open("r", encoding="utf-8") as f:
        creds = yaml.safe_load(f)

    username = creds.get("username")
    password = creds.get("password")

    if not username or not password:
        raise CopernicusAPIError(
            "Credentials file must contain 'username' and 'password' fields."
        )

    logger.debug(f"Generating fresh token for user: {username}")

    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    try:
        response = requests.post(token_url, data=data, timeout=300)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")

        if not access_token:
            raise CopernicusAPIError("No access_token in response")

        logger.info("Successfully generated fresh access token")
        return access_token

    except requests.RequestException as e:
        raise CopernicusAPIError(f"Failed to generate token: {e}") from e


def _read_token(token_path: TokenSource = None) -> str:
    """Get an OAuth token for Copernicus API access.

    First attempts to generate a fresh token from .credentials.yaml.
    Falls back to reading a static token from file if credentials are not available.

    Parameters
    ----------
    token_path:
        Optional path to a file containing a static token. When omitted,
        first tries .credentials.yaml, then .copernicus_token.txt.

    Returns
    -------
    str:
        Valid access token.
    """
    # Try to generate fresh token from credentials first
    credentials_path = Path(".credentials.yaml")
    if credentials_path.exists():
        try:
            return _generate_token_from_credentials(credentials_path)
        except CopernicusAPIError as e:
            logger.warning(f"Failed to generate token from credentials: {e}")
            logger.warning("Falling back to static token file")

    # Fall back to static token file
    if token_path is None:
        token_path = Path(".copernicus_token.txt")
    elif not isinstance(token_path, Path):
        token_path = Path(token_path)

    if not token_path.exists():
        raise CopernicusAPIError(
            f"Token file not found at '{token_path}'. Please create either:\n"
            "1. .credentials.yaml with username and password (recommended), or\n"
            "2. .copernicus_token.txt with a valid access token"
        )

    logger.warning("Using static token from file (may be expired)")
    return token_path.read_text(encoding="utf-8").strip()


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


def _select_asset(
    feature: Mapping[str, Any],
    asset_type: AssetType = "PRODUCT"
) -> Tuple[str, Mapping[str, Any]]:
    """Pick asset from a STAC feature based on desired type.

    Parameters
    ----------
    feature:
        STAC feature containing assets.
    asset_type:
        Type of asset to select. Either "PRODUCT" (full satellite data archive)
        or "QUICKLOOK" (preview image). Default is "PRODUCT".

    Returns
    -------
    tuple:
        (asset_name, asset_info) for the selected asset.
    """

    assets: MutableMapping[str, Mapping[str, Any]] = feature.get("assets", {})  # type: ignore[assignment]

    if asset_type == "PRODUCT":
        # Priority: PRODUCT (full satellite data archive)
        if "PRODUCT" in assets:
            return "PRODUCT", assets["PRODUCT"]

        # Fallback to any available asset
        if assets:
            key = next(iter(assets.keys()))
            logger.warning(f"PRODUCT not found, using '{key}' instead")
            return key, assets[key]

    elif asset_type == "QUICKLOOK":
        # Priority: QUICKLOOK/quicklook preview images
        quicklook_keys = ("QUICKLOOK", "quicklook", "thumbnail", "visual", "rendered_preview")
        for key in quicklook_keys:
            if key in assets:
                return key, assets[key]

        # Fallback to any available asset
        if assets:
            key = next(iter(assets.keys()))
            logger.warning(f"QUICKLOOK not found, using '{key}' instead")
            return key, assets[key]

    raise CopernicusAPIError(
        f"No downloadable asset found in feature (requested: {asset_type})"
    )


def _search_features(
    bbox: BBox,
    datetime_range: Any,
    collection: Collection = "SENTINEL-2",
    limit: int = 1,
    order: str = "desc",
    raise_on_empty: bool = True,
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

    # STAC search endpoint is publicly accessible and does not require authentication
    response = requests.post(STAC_SEARCH_URL, headers={"Content-Type": "application/json"}, json=payload, timeout=1000)
    if response.status_code != 200:
        raise CopernicusAPIError(
            "Copernicus STAC search failed with status "
            f"{response.status_code}: {response.text}"
        )

    body = response.json()
    features = body.get("features", [])
    if not features and raise_on_empty:
        raise CopernicusAPIError("No features returned for the given parameters.")

    return features


def _download_asset(href: str, token: str) -> bytes:
    """Download a small asset from Copernicus catalog into memory.

    For small assets like thumbnails/previews only. Use _download_asset_to_file
    for large PRODUCT downloads.
    """
    timeout = 300
    logger.debug(f"Downloading asset from {href} (timeout={timeout}s)")

    response = requests.get(
        href,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout
    )

    if response.status_code != 200:
        raise CopernicusAPIError(
            f"Failed to download asset from {href}: {response.status_code} {response.text}"
        )

    logger.debug(f"Downloaded {len(response.content)} bytes")
    return response.content


def _download_asset_to_file(href: str, token: str, output_path: Path) -> int:
    """Download a large asset from Copernicus catalog directly to disk.

    Streams the download to avoid loading large files (800MB+) into memory.
    Handles redirects manually to preserve Authorization headers.

    Returns
    -------
    int:
        Number of bytes downloaded.
    """
    # PRODUCT files can be several GB, so use longer timeout
    timeout = 900

    logger.debug(f"Streaming download from {href} to {output_path}")

    # For PRODUCT downloads, check for redirect and handle manually
    # (requests may strip auth headers on cross-domain redirects)
    if "/Products(" in href:
        # First request: check for redirect (don't follow)
        check_response = requests.get(
            href,
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=False,
            timeout=300
        )

        if check_response.status_code in (301, 302, 303, 307, 308):
            # Handle redirect manually with auth header
            redirect_url = check_response.headers.get("Location")
            logger.debug(f"Following redirect to: {redirect_url}")
            href = redirect_url

    # Stream download to file
    response = requests.get(
        href,
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=timeout
    )

    if response.status_code != 200:
        raise CopernicusAPIError(
            f"Failed to download asset from {href}: {response.status_code} {response.text}"
        )

    content_length = int(response.headers.get("Content-Length", 0))
    logger.info(f"Streaming {content_length} bytes to {output_path.name}")

    # Stream download in chunks to avoid memory issues
    bytes_downloaded = 0
    chunk_size = 8192  # 8KB chunks

    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)

                # Log progress for large files
                if bytes_downloaded % (100 * 1024 * 1024) == 0:  # Every 100MB
                    progress_mb = bytes_downloaded / (1024 * 1024)
                    logger.info(f"  Progress: {progress_mb:.1f} MB downloaded")

    logger.info(f"Download complete: {bytes_downloaded} bytes")
    return bytes_downloaded

def _select_best_asset(feature):
    assets = feature.get("assets", {})
    # Prefer a ready true-color or 10 m bands
    preferred_order = ["visual", "TCI", "orthoimage"]  # names vary by catalog
    # 1) try visual/TCI if high-quality JP2/COG
    for name in preferred_order:
        a = assets.get(name)
        if a and a.get("type", "").startswith(("image/jp2", "image/tiff")) \
           and "overview" not in (a.get("roles") or []):
            return name, a

    # 2) else fall back to building true-color from 10 m bands
    band_names = ["B04", "B03", "B02"]  # R,G,B at 10 m
    if all(b in assets for b in band_names):
        # return one for now; you can download all three and compose RGB locally
        return "B04_B03_B02", {  # sentinel marker so you know to fetch 3 assets
            "hrefs": [assets[b]["href"] for b in band_names],
            "type": "image/jp2",
            "roles": ["data", "composite"]
        }

    # 3) last resort: pick any 10 m band
    for b in ["B04", "B03", "B02", "B08"]:
        a = assets.get(b)
        if a and a.get("type","").startswith("image/jp2"):
            return b, a

    # Never use quicklook unless user explicitly asks for a preview
    raise CopernicusAPIError("No high-quality data assets found.")


def fetch_image_for_bbox(
    bbox: BBox,
    date: Any,
    *,
    token_path: TokenSource = None,
    collection: Collection = "SENTINEL-2",
    limit: int = 50,
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
    features = _search_features(bbox=bbox, datetime_range=date, collection=collection, limit=limit)
    # feature = next(iter(features))
    feature = min(features, key=lambda f: f["properties"].get("cloudCover", 1000))

    asset_name, asset_info = _select_best_asset(feature)
    if asset_name == "B04_B03_B02":
        hrefs = asset_info["hrefs"]
        # download 3 JP2s and compose RGB locally; save as GeoTIFF/COG if you wish
        # content = _download_and_compose_rgb(hrefs, token)
        mime_type = "image/tiff; application=geotiff"
    else:
        href = asset_info.get("href")
        if not href:
            raise CopernicusAPIError(f"Asset '{asset_name}' had no href.")
        content = _download_asset(href, token)
        mime_type = asset_info.get("type") or "application/octet-stream"


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
        "cloud_cover": feature.get("properties", {}).get("cloudCover"),
        "collection": collection,
        "feature_id": feature.get("id"),
        "asset_name": asset_name,
        "asset_type": asset_info.get("type"),
    }

    mime_type = asset_info.get("type") or "application/octet-stream"

    return CopernicusImage(content=content, metadata=metadata, asset_href=href if asset_name!="B04_B03_B02" else hrefs, mime_type=mime_type)


def fetch_latest_image_for_bbox(
    bbox: BBox,
    *,
    target_date: dt.datetime | None = None,
    lookback_days: int = 30,
    max_cloud_cover: float = 30.0,
    token_path: TokenSource = None,
    collection: Collection = "SENTINEL-2",
) -> CopernicusImage:
    """Retrieve the latest high-quality image for a bounding box.

    Searches for images within a fixed time window and returns the most recent one
    that meets the quality criteria (cloud cover threshold).

    Parameters
    ----------
    bbox:
        Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
    target_date:
        End date for the search window. Defaults to current UTC time.
    lookback_days:
        Number of days to look back from target_date. Default is 30 days (1 month).
    max_cloud_cover:
        Maximum acceptable cloud cover percentage (0-100). Default is 30%.
        Images with higher cloud cover are excluded.
    token_path:
        Optional path to a file containing the access token.
    collection:
        STAC collection to query. Default is "SENTINEL-2".

    Returns
    -------
    CopernicusImage:
        The most recent image that meets the quality criteria.

    Raises
    ------
    CopernicusAPIError:
        If no suitable image is found within the search window.

    Examples
    --------
    >>> # Search for latest image within last 30 days with <30% cloud cover
    >>> image = fetch_latest_image_for_bbox(
    ...     bbox=(lon_min, lat_min, lon_max, lat_max)
    ... )

    >>> # Stricter quality requirements
    >>> image = fetch_latest_image_for_bbox(
    ...     bbox=(lon_min, lat_min, lon_max, lat_max),
    ...     lookback_days=60,
    ...     max_cloud_cover=10.0
    ... )
    """
    if target_date is None:
        target_date = dt.datetime.now(dt.timezone.utc)
    elif target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=dt.timezone.utc)

    start = target_date - dt.timedelta(days=lookback_days)
    datetime_range = (start, target_date)

    logger.info(
        f"Searching for images in bbox {bbox} from {start.date()} to {target_date.date()} "
        f"(max_cloud_cover={max_cloud_cover}%)"
    )

    token = _read_token(token_path)

    # Search for multiple features to allow quality filtering
    features = _search_features(
        bbox=bbox,
        datetime_range=datetime_range,
        collection=collection,
        limit=50,
        order="desc",
        raise_on_empty=False,
    )

    if not features:
        raise CopernicusAPIError(
            f"No images found in bbox {bbox} from {start.date()} to {target_date.date()}"
        )

    # Filter by cloud cover
    suitable_features = [
        f for f in features
        if f.get("properties", {}).get("cloudCover", 100) <= max_cloud_cover
    ]

    if not suitable_features:
        min_cloud = min(f.get("properties", {}).get("cloudCover", 100) for f in features)
        raise CopernicusAPIError(
            f"Found {len(features)} images but none with cloud cover <= {max_cloud_cover}% "
            f"(minimum found: {min_cloud:.1f}%)"
        )

    # Select the most recent image with lowest cloud cover
    # Sort by datetime (most recent first), then by cloud cover (lowest first)
    def sort_key(f: Mapping[str, Any]) -> Tuple[str, float]:
        props = f.get("properties", {})
        timestamp_str = props.get("datetime", "")
        cloud_cover = props.get("cloudCover", 100)
        return (timestamp_str, cloud_cover)

    best_feature = max(suitable_features, key=sort_key)
    cloud_cover = best_feature.get("properties", {}).get("cloudCover", "N/A")
    timestamp = best_feature.get("properties", {}).get("datetime", "N/A")

    logger.info(
        f"Selected image: timestamp={timestamp}, cloud_cover={cloud_cover}%"
    )

    # Download the best feature
    asset_name, asset_info = _select_asset(best_feature)
    href = asset_info.get("href")
    if not href:
        raise CopernicusAPIError(
            f"Asset '{asset_name}' did not contain a download link (href missing)."
        )

    content = _download_asset(href, token)
    metadata = {
        "bbox": bbox,
        "timestamp": timestamp,
        "cloud_cover": cloud_cover,
        "collection": collection,
        "feature_id": best_feature.get("id"),
        "asset_name": asset_name,
        "asset_type": asset_info.get("type"),
    }

    mime_type = asset_info.get("type") or "application/octet-stream"

    return CopernicusImage(
        content=content,
        metadata=metadata,
        asset_href=href,
        mime_type=mime_type
    )


def fetch_all_images_in_period(
    bbox: BBox,
    start_date: dt.datetime | dt.date,
    end_date: dt.datetime | dt.date,
    *,
    max_cloud_cover: float = 30.0,
    token_path: TokenSource = None,
    collection: Collection = "SENTINEL-2",
    asset_type: AssetType = "PRODUCT",
    output_dir: Path | None = None,
    order: str = "asc",
):
    """Generator that yields download info for high-quality images in a date range.

    Searches for all images within the specified time period and yields download
    information for those that meet quality criteria (cloud cover threshold).
    This avoids loading large files into memory.

    The search limit is automatically set to the number of days in the interval,
    which is sufficient for most satellite collections (e.g., Sentinel-2 has ~5 day
    revisit time).

    Parameters
    ----------
    bbox:
        Bounding box defined as (min_lon, min_lat, max_lon, max_lat).
    start_date:
        Start date of the search period (inclusive).
    end_date:
        End date of the search period (inclusive).
    max_cloud_cover:
        Maximum acceptable cloud cover percentage (0-100). Default is 30%.
        Images with higher cloud cover are excluded.
    token_path:
        Optional path to a file containing the access token.
    collection:
        STAC collection to query. Default is "SENTINEL-2".
    asset_type:
        Type of asset to download. Either "PRODUCT" (full satellite archive)
        or "QUICKLOOK" (preview image). Default is "PRODUCT".
    output_dir:
        Deprecated, no longer used.
    order:
        Sort order for results. Either "asc" (chronological) or "desc" (reverse
        chronological). Default is "asc".

    Yields
    ------
    dict:
        Dictionary with keys: 'href', 'metadata', 'extension', 'token_path',
        'index', 'total'. Caller should download using _download_asset_to_file.

    Examples
    --------
    >>> # Stream downloads one by one
    >>> for item in fetch_all_images_in_period(
    ...     bbox=(lon_min, lat_min, lon_max, lat_max),
    ...     start_date=dt.date(2024, 1, 1),
    ...     end_date=dt.date(2024, 3, 31)
    ... ):
    ...     # Download to file without loading into memory
    ...     token = _read_token(item['token_path'])
    ...     _download_asset_to_file(item['href'], token, output_path)
    """
    # Convert dates to datetime if needed
    if isinstance(start_date, dt.date) and not isinstance(start_date, dt.datetime):
        start_date = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.timezone.utc)
    elif isinstance(start_date, dt.datetime) and start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=dt.timezone.utc)

    if isinstance(end_date, dt.date) and not isinstance(end_date, dt.datetime):
        end_date = dt.datetime.combine(end_date, dt.time.max, tzinfo=dt.timezone.utc)
    elif isinstance(end_date, dt.datetime) and end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=dt.timezone.utc)

    datetime_range = (start_date, end_date)

    # Calculate number of days in the interval
    days_in_interval = (end_date - start_date).days + 1  # +1 to include end date

    logger.info(
        f"Searching for all images in bbox {bbox} from {start_date.date()} to {end_date.date()} "
        f"({days_in_interval} days, max_cloud_cover={max_cloud_cover}%)"
    )

    # Use number of days as limit (Sentinel-2 revisit time is ~5 days, so this is generous)
    # Note: STAC API may have its own maximum limit
    features = _search_features(
        bbox=bbox,
        datetime_range=datetime_range,
        collection=collection,
        limit=days_in_interval,
        order=order,
        raise_on_empty=False,
    )

    if not features:
        logger.warning(f"No images found in bbox {bbox} from {start_date.date()} to {end_date.date()}")
        return []

    # Filter by cloud cover
    suitable_features = [
        f for f in features
        if f.get("properties", {}).get("cloudCover", 100) <= max_cloud_cover
    ]

    if not suitable_features:
        min_cloud = min(f.get("properties", {}).get("cloudCover", 100) for f in features)
        logger.warning(
            f"Found {len(features)} images but none with cloud cover <= {max_cloud_cover}% "
            f"(minimum found: {min_cloud:.1f}%)"
        )
        return []

    logger.info(f"Found {len(suitable_features)} suitable images (out of {len(features)} total)")

    # Yield image metadata for streaming downloads (generator pattern)
    for i, feature in enumerate(suitable_features, 1):
        cloud_cover = feature.get("properties", {}).get("cloudCover", "N/A")
        timestamp = feature.get("properties", {}).get("datetime", "N/A")

        logger.info(f"Processing image {i}/{len(suitable_features)}: {timestamp}, cloud_cover={cloud_cover}%")

        try:
            asset_name, asset_info = _select_asset(feature, asset_type=asset_type)
            href = asset_info.get("href")
            if not href:
                logger.warning(f"Skipping image {i}: no download link found")
                continue

            # Determine file extension
            mime_type = asset_info.get("type") or "application/octet-stream"
            if asset_name == "PRODUCT":
                extension = ".zip"
            else:
                extension = mimetypes.guess_extension(mime_type) or ".bin"

            metadata = {
                "bbox": bbox,
                "timestamp": timestamp,
                "cloud_cover": cloud_cover,
                "collection": collection,
                "feature_id": feature.get("id"),
                "asset_name": asset_name,
                "asset_type": mime_type,
            }

            # Yield download info for caller to handle
            yield {
                "href": href,
                "metadata": metadata,
                "extension": extension,
                "token_path": token_path,
                "index": i,
                "total": len(suitable_features),
            }

        except Exception as e:
            logger.warning(f"Failed to process image {i} ({timestamp}): {e}")
            continue


def serialize_metadata(image: CopernicusImage) -> str:
    """Serialize metadata to JSON for persistence."""

    return json.dumps(image.metadata, indent=2, sort_keys=True)


__all__ = [
    "AssetType",
    "BBox",
    "Collection",
    "CopernicusImage",
    "CopernicusAPIError",
    "fetch_image_for_bbox",
    "fetch_latest_image_for_bbox",
    "fetch_all_images_in_period",
    "serialize_metadata",
]


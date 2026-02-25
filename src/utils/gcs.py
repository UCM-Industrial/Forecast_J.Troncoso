"""RECAST — Google Cloud Storage utilities.

Provides upload/download helpers with a **local filesystem fallback**
when ``STORAGE_BACKEND=local`` (for development without GCP credentials).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from google.cloud.storage import Blob

logger = get_logger("gcs")


# ── Public API ───────────────────────────────────────────────


def upload_blob(
    local_path: str | Path,
    remote_path: str,
    *,
    bucket_name: str | None = None,
) -> str:
    """Upload a local file to GCS (or local fallback).

    Args:
        local_path: Path to the file on disk.
        remote_path: Destination path inside the bucket / local dir.
        bucket_name: Override bucket from config.

    Returns:
        The URI of the uploaded file (``gs://...`` or local path).
    """
    settings = get_settings()
    local_path = Path(local_path)

    if settings.storage_backend == "local":
        return _local_upload(local_path, remote_path, settings.local_data_dir)

    return _gcs_upload(local_path, remote_path, bucket_name or settings.gcs.bucket)


def download_blob(
    remote_path: str,
    local_path: str | Path,
    *,
    bucket_name: str | None = None,
) -> Path:
    """Download a file from GCS (or local fallback) to disk.

    Args:
        remote_path: Source path inside the bucket / local dir.
        local_path: Destination path on disk.
        bucket_name: Override bucket from config.

    Returns:
        Path to the downloaded file.
    """
    settings = get_settings()
    local_path = Path(local_path)

    if settings.storage_backend == "local":
        return _local_download(remote_path, local_path, settings.local_data_dir)

    return _gcs_download(remote_path, local_path, bucket_name or settings.gcs.bucket)


def list_blobs(
    prefix: str,
    *,
    bucket_name: str | None = None,
) -> list[str]:
    """List files under a prefix in GCS (or local fallback).

    Args:
        prefix: Path prefix to filter by.
        bucket_name: Override bucket from config.

    Returns:
        List of relative paths matching the prefix.
    """
    settings = get_settings()

    if settings.storage_backend == "local":
        return _local_list(prefix, settings.local_data_dir)

    return _gcs_list(prefix, bucket_name or settings.gcs.bucket)


def blob_exists(
    remote_path: str,
    *,
    bucket_name: str | None = None,
) -> bool:
    """Check if a blob exists in GCS (or local fallback).

    Args:
        remote_path: Path inside the bucket / local dir.
        bucket_name: Override bucket from config.

    Returns:
        ``True`` if the file exists.
    """
    settings = get_settings()

    if settings.storage_backend == "local":
        base = Path(settings.local_data_dir)
        return (base / remote_path).exists()

    return _gcs_exists(remote_path, bucket_name or settings.gcs.bucket)


# ── GCS implementation ───────────────────────────────────────


def _get_bucket(bucket_name: str):
    """Lazily import and return a GCS bucket object."""
    from google.cloud import storage  # noqa: PLC0415

    client = storage.Client()
    return client.bucket(bucket_name)


def _gcs_upload(local_path: Path, remote_path: str, bucket_name: str) -> str:
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{bucket_name}/{remote_path}"
    logger.info("Uploaded %s → %s", local_path.name, uri)
    return uri


def _gcs_download(remote_path: str, local_path: Path, bucket_name: str) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(remote_path)
    blob.download_to_filename(str(local_path))
    logger.info("Downloaded gs://%s/%s → %s", bucket_name, remote_path, local_path)
    return local_path


def _gcs_list(prefix: str, bucket_name: str) -> list[str]:
    bucket = _get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [b.name for b in blobs]


def _gcs_exists(remote_path: str, bucket_name: str) -> bool:
    bucket = _get_bucket(bucket_name)
    blob = bucket.blob(remote_path)
    return blob.exists()


# ── Local filesystem fallback ────────────────────────────────


def _local_upload(local_path: Path, remote_path: str, data_dir: str) -> str:
    dest = Path(data_dir) / remote_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest)
    logger.info("Copied %s → %s (local)", local_path.name, dest)
    return str(dest)


def _local_download(remote_path: str, local_path: Path, data_dir: str) -> Path:
    src = Path(data_dir) / remote_path
    if not src.exists():
        msg = f"Local file not found: {src}"
        raise FileNotFoundError(msg)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, local_path)
    logger.info("Copied %s → %s (local)", src, local_path)
    return local_path


def _local_list(prefix: str, data_dir: str) -> list[str]:
    base = Path(data_dir)
    target = base / prefix
    if not target.exists():
        return []
    return [str(p.relative_to(base)) for p in target.rglob("*") if p.is_file()]

"""Utilities for staging model artifacts from Google Cloud Storage."""

from __future__ import annotations

import logging
from pathlib import Path

from google.cloud import storage

from app.config import Config


logger = logging.getLogger(__name__)


class ModelDownloadError(RuntimeError):
    """Raised when required model artifacts cannot be staged locally."""


def ensure_modernbert_models(
    component_model_dir: str | Path | None = None,
    relation_model_dir: str | Path | None = None,
) -> None:
    """Ensure the local ModernBERT model directories exist.

    In local development this is a no-op when the checked-out ``models/``
    directory is present. If a deployment starts without local model files,
    the first ModernBERT use downloads them from the configured GCS bucket
    into the configured model directories.
    """
    component_dir = Path(component_model_dir or Config.COMPONENT_MODEL_DIR)
    relation_dir = Path(relation_model_dir or Config.RELATION_MODEL_DIR)
    required_dirs = (component_dir, relation_dir)

    if all(_has_required_files(path) for path in required_dirs):
        return

    if not Config.MODEL_GCS_BUCKET:
        missing = ", ".join(str(path) for path in required_dirs if not _has_required_files(path))
        raise ModelDownloadError(
            "ModernBERT model files are missing and MODEL_GCS_BUCKET is not set: "
            f"{missing}"
        )

    for path in required_dirs:
        path.mkdir(parents=True, exist_ok=True)

    client = storage.Client(project=Config.GOOGLE_CLOUD_PROJECT or None)
    bucket = client.bucket(Config.MODEL_GCS_BUCKET)

    _download_prefix(bucket, Config.COMPONENT_MODEL_GCS_PREFIX, component_dir)
    _download_prefix(bucket, Config.RELATION_MODEL_GCS_PREFIX, relation_dir)

    for path in required_dirs:
        if not _has_required_files(path):
            raise ModelDownloadError(f"Downloaded model directory is incomplete: {path}")


def _has_required_files(path: Path) -> bool:
    """Return True when a Hugging Face sequence-classification model is present."""
    return all((path / name).exists() for name in Config.MODEL_REQUIRED_FILES)


def _download_prefix(bucket: storage.Bucket, prefix: str, destination: Path) -> None:
    """Download all objects under *prefix* into *destination*."""
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix:
        raise ModelDownloadError("Model GCS prefixes must not be empty.")

    marker = destination / ".gcs-download-complete"
    if marker.exists() and _has_required_files(destination):
        return

    blobs = list(bucket.list_blobs(prefix=f"{normalized_prefix}/"))
    files = [blob for blob in blobs if not blob.name.endswith("/")]
    if not files:
        raise ModelDownloadError(
            f"No model files found at gs://{bucket.name}/{normalized_prefix}/"
        )

    logger.info(
        "Downloading %s model files from gs://%s/%s/ to %s",
        len(files),
        bucket.name,
        normalized_prefix,
        destination,
    )
    for blob in files:
        relative_name = blob.name.removeprefix(f"{normalized_prefix}/")
        target = destination / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))

    marker.write_text("ok\n", encoding="utf-8")

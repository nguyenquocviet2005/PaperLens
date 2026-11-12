"""Storage backends for PaperLens artifacts."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base storage failure."""


class PathTraversalError(StorageError):
    """Raised when a key escapes the storage root."""


class ObjectNotFoundError(StorageError):
    """Raised when an object does not exist."""


def _normalize_key(key: str) -> str:
    cleaned = key.replace("\\", "/").lstrip("/")
    parts = [part for part in cleaned.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise PathTraversalError(f"Path traversal rejected for key: {key!r}")
    return "/".join(parts)


class StorageBackend(ABC):
    """Common storage interface."""

    @abstractmethod
    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def save_text(self, key: str, text: str, encoding: str = "utf-8") -> str:
        raise NotImplementedError

    @abstractmethod
    def save_json(self, key: str, payload: Any) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    def read_text(self, key: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(key).decode(encoding)

    def read_json(self, key: str) -> Any:
        return json.loads(self.read_text(key))

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def delete_object(self, key: str) -> None:
        raise NotImplementedError


class LocalStorage(StorageBackend):
    """Filesystem storage with atomic writes."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        normalized = _normalize_key(key)
        path = (self.root / normalized).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise PathTraversalError(f"Path traversal rejected for key: {key!r}") from exc
        return path

    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp_name).replace(path)
        except Exception:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return _normalize_key(key)

    def save_text(self, key: str, text: str, encoding: str = "utf-8") -> str:
        return self.save_bytes(key, text.encode(encoding), content_type="text/plain")

    def save_json(self, key: str, payload: Any) -> str:
        body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return self.save_text(key, body, encoding="utf-8")

    def read_bytes(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise ObjectNotFoundError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def list_objects(self, prefix: str = "") -> list[str]:
        normalized = _normalize_key(prefix) if prefix else ""
        base = self.root / normalized if normalized else self.root
        if not base.exists():
            return []
        results: list[str] = []
        for path in base.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self.root).as_posix()
                results.append(rel)
        return sorted(results)

    def delete_object(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            path.unlink()


class S3Storage(StorageBackend):
    """Amazon S3 backend using the default AWS credential chain (IAM task role in ECS)."""

    def __init__(self, bucket_name: str, region: str | None = None, client: Any | None = None) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise StorageError("boto3 is required for S3 backend") from exc
            client = boto3.client("s3", region_name=region)

        self.bucket_name = bucket_name
        self.client = client

    @staticmethod
    def _is_missing(exc: Exception) -> bool:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def save_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        normalized = _normalize_key(key)
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=normalized,
            Body=data,
            ContentType=content_type or "application/octet-stream",
        )
        return normalized

    def save_text(self, key: str, text: str, encoding: str = "utf-8") -> str:
        return self.save_bytes(key, text.encode(encoding), content_type="text/plain; charset=utf-8")

    def save_json(self, key: str, payload: Any) -> str:
        body = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        return self.save_bytes(
            key,
            body.encode("utf-8"),
            content_type="application/json; charset=utf-8",
        )

    def read_bytes(self, key: str) -> bytes:
        normalized = _normalize_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=normalized)
        except Exception as exc:
            if self._is_missing(exc):
                raise ObjectNotFoundError(key) from exc
            raise
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=_normalize_key(key))
        except Exception as exc:
            if self._is_missing(exc):
                return False
            raise
        return True

    def list_objects(self, prefix: str = "") -> list[str]:
        normalized = _normalize_key(prefix) if prefix else ""
        paginator = self.client.get_paginator("list_objects_v2")
        results: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=normalized):
            results.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(results)

    def delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=_normalize_key(key))


def get_storage(settings: Settings | None = None) -> StorageBackend:
    cfg = settings or get_settings()
    if cfg.storage_backend == "s3":
        if not cfg.s3_bucket_name:
            raise StorageError("S3_BUCKET_NAME is required when STORAGE_BACKEND=s3")
        logger.info("Using S3 storage backend bucket=%s", cfg.s3_bucket_name)
        return S3Storage(bucket_name=cfg.s3_bucket_name, region=cfg.aws_region)
    root = cfg.local_storage_root
    if not root.is_absolute():
        root = Path.cwd() / root
    logger.info("Using local storage backend root=%s", root)
    return LocalStorage(root)


# Logical object key helpers -------------------------------------------------

def paper_raw_pdf_key(paper_id: str) -> str:
    return f"raw/papers/{paper_id}/source.pdf"


def paper_parsed_key(paper_id: str, name: str) -> str:
    return f"parsed/papers/{paper_id}/{name}"


def paper_normalized_key(paper_id: str, name: str) -> str:
    return f"normalized/papers/{paper_id}/{name}"


def paper_asset_key(paper_id: str, kind: str, name: str) -> str:
    return f"assets/papers/{paper_id}/{kind}/{name}"


def paper_enrichment_key(paper_id: str, kind: str, name: str) -> str:
    return f"enrichment/papers/{paper_id}/{kind}/{name}"


def paper_meta_key(paper_id: str) -> str:
    return f"normalized/papers/{paper_id}/meta.json"

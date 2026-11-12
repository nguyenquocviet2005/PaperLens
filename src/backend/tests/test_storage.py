"""Unit tests for local and S3 storage backends."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.infrastructure.storage import (
    LocalStorage,
    ObjectNotFoundError,
    PathTraversalError,
    S3Storage,
    StorageError,
    get_storage,
)


def test_save_and_read_bytes(tmp_path):
    store = LocalStorage(tmp_path)
    key = store.save_bytes("raw/papers/abc/source.pdf", b"%PDF-1.4 test")
    assert key == "raw/papers/abc/source.pdf"
    assert store.read_bytes(key) == b"%PDF-1.4 test"
    assert store.exists(key)


def test_json_round_trip(tmp_path):
    store = LocalStorage(tmp_path)
    key = store.save_json("normalized/papers/abc/meta.json", {"ok": True, "n": 1})
    assert store.read_json(key) == {"ok": True, "n": 1}


def test_path_traversal_rejected(tmp_path):
    store = LocalStorage(tmp_path)
    with pytest.raises(PathTraversalError):
        store.save_text("../secret.txt", "nope")
    with pytest.raises(PathTraversalError):
        store.read_bytes("foo/../../etc/passwd")


def test_list_and_delete(tmp_path):
    store = LocalStorage(tmp_path)
    store.save_text("a/b.txt", "hello")
    store.save_text("a/c.txt", "world")
    listed = store.list_objects("a")
    assert "a/b.txt" in listed
    assert "a/c.txt" in listed
    store.delete_object("a/b.txt")
    assert not store.exists("a/b.txt")


@pytest.fixture
def s3_store(monkeypatch):
    moto = pytest.importorskip("moto")
    import boto3

    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(name, "testing")
    with moto.mock_aws():
        client = boto3.client("s3", region_name="ap-northeast-1")
        client.create_bucket(
            Bucket="paperlens-test",
            CreateBucketConfiguration={"LocationConstraint": "ap-northeast-1"},
        )
        yield S3Storage("paperlens-test", client=client)


def test_s3_save_and_read_bytes(s3_store):
    key = s3_store.save_bytes("/raw/papers/abc/source.pdf", b"%PDF-1.4 test", "application/pdf")
    assert key == "raw/papers/abc/source.pdf"
    assert s3_store.read_bytes(key) == b"%PDF-1.4 test"
    assert s3_store.exists(key)
    head = s3_store.client.head_object(Bucket="paperlens-test", Key=key)
    assert head["ContentType"] == "application/pdf"


def test_s3_json_round_trip(s3_store):
    key = s3_store.save_json("normalized/papers/abc/meta.json", {"ok": True, "n": 1})
    assert s3_store.read_json(key) == {"ok": True, "n": 1}


def test_s3_missing_object(s3_store):
    assert not s3_store.exists("missing.txt")
    with pytest.raises(ObjectNotFoundError):
        s3_store.read_bytes("missing.txt")


def test_s3_path_traversal_rejected(s3_store):
    with pytest.raises(PathTraversalError):
        s3_store.save_text("../secret.txt", "nope")


def test_s3_list_and_delete(s3_store):
    s3_store.save_text("a/b.txt", "hello")
    s3_store.save_text("a/c.txt", "world")
    s3_store.save_text("other/d.txt", "x")
    assert s3_store.list_objects("a") == ["a/b.txt", "a/c.txt"]
    s3_store.delete_object("a/b.txt")
    s3_store.delete_object("a/never-existed.txt")
    assert not s3_store.exists("a/b.txt")


def test_get_storage_s3_requires_bucket():
    with pytest.raises(StorageError):
        get_storage(Settings(storage_backend="s3", s3_bucket_name=""))

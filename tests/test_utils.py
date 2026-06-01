"""Tests for utility functions."""
from __future__ import annotations

import pytest

from golf_mapper.utils import DiskCache, ensure_dir, hash_dict, retry, set_seeds


def test_retry_succeeds_after_failures():
    calls: list[int] = []

    @retry(max_retries=3, backoff_base=0.001, jitter=False)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_retry_exhausted_raises():
    @retry(max_retries=2, backoff_base=0.001, jitter=False)
    def always_fails() -> None:
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="always"):
        always_fails()


def test_retry_only_catches_specified_exception():
    @retry(max_retries=2, backoff_base=0.001, exceptions=(ValueError,))
    def raises_type_error() -> None:
        raise TypeError("not caught")

    with pytest.raises(TypeError):
        raises_type_error()


def test_disk_cache_roundtrip_pickle(tmp_path):
    cache = DiskCache(tmp_path / "cache", serializer="pickle")
    cache.set("key1", {"x": 1, "y": [2, 3]})
    assert cache.has("key1")
    assert cache.get("key1") == {"x": 1, "y": [2, 3]}


def test_disk_cache_roundtrip_json(tmp_path):
    cache = DiskCache(tmp_path / "cache", serializer="json")
    cache.set("key2", [1, 2, 3])
    assert cache.get("key2") == [1, 2, 3]


def test_disk_cache_miss_raises(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    with pytest.raises(KeyError):
        cache.get("missing")


def test_disk_cache_has_returns_false_for_missing(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    assert not cache.has("ghost")


def test_disk_cache_decorator(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    calls: list[int] = []

    @cache.cached(key_fn=lambda x: f"double:{x}")
    def double(x: int) -> int:
        calls.append(x)
        return x * 2

    assert double(5) == 10
    assert double(5) == 10   # served from cache
    assert len(calls) == 1   # function body executed only once
    assert double(7) == 14   # different key — executes again
    assert len(calls) == 2


def test_disk_cache_invalid_serializer(tmp_path):
    with pytest.raises(ValueError):
        DiskCache(tmp_path / "cache", serializer="xml")


def test_set_seeds_does_not_raise():
    set_seeds(42)
    set_seeds(0)


def test_hash_dict_is_stable():
    d = {"a": 1, "b": [2, 3]}
    assert hash_dict(d) == hash_dict(d)
    assert len(hash_dict(d)) == 16


def test_hash_dict_differs_on_different_input():
    assert hash_dict({"a": 1}) != hash_dict({"a": 2})


def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()
    assert result == target


def test_ensure_dir_idempotent(tmp_path):
    target = tmp_path / "x"
    ensure_dir(target)
    ensure_dir(target)  # second call must not raise
    assert target.is_dir()

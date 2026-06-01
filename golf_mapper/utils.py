"""Logging, caching, retry, and seed utilities.

No bare print() calls in this module — all output goes through logging.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_ROOT_CONFIGURED = False


def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    _ROOT_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seeds(seed: int = 42) -> None:
    """Deterministically seed Python random, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def retry(
    max_retries: int = 5,
    backoff_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry on specified exceptions with exponential backoff + jitter.

    Usage::

        @retry(max_retries=3, backoff_base=2.0, exceptions=(requests.HTTPError,))
        def fetch(url):
            ...
    """
    _log = get_logger("utils.retry")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_retries:
                        _log.error(
                            "%s failed after %d retries: %s",
                            func.__name__, max_retries, exc,
                        )
                        raise
                    delay = backoff_base ** attempt
                    if jitter:
                        delay *= 1.0 + random.random() * 0.25
                    _log.warning(
                        "%s attempt %d/%d failed (%s). Retrying in %.1fs…",
                        func.__name__, attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)

        return wrapper  # type: ignore[return-value]

    return decorator


class DiskCache:
    """File-based cache that serializes values as pickle (default) or JSON.

    Keys are hashed with SHA-256 so any string can be used safely as a filesystem key.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        serializer: str = "pickle",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if serializer not in ("pickle", "json"):
            raise ValueError(f"serializer must be 'pickle' or 'json', got {serializer!r}")
        self.serializer = serializer
        self._suffix = ".pkl" if serializer == "pickle" else ".json"
        self._log = get_logger("utils.DiskCache")

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / (digest + self._suffix)

    def has(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str) -> Any:
        p = self._path(key)
        if not p.exists():
            raise KeyError(key)
        mode = "rb" if self.serializer == "pickle" else "r"
        with p.open(mode) as fh:
            data = pickle.load(fh) if self.serializer == "pickle" else json.load(fh)
        self._log.debug("Cache hit: %.80s", key)
        return data

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        mode = "wb" if self.serializer == "pickle" else "w"
        with p.open(mode) as fh:
            if self.serializer == "pickle":
                pickle.dump(value, fh)
            else:
                json.dump(value, fh, ensure_ascii=False, indent=2)
        self._log.debug("Cache set: %.80s → %s", key, p.name)

    def cached(self, key_fn: Callable[..., str]) -> Callable[[F], F]:
        """Decorator: cache the return value, keyed by ``key_fn(*args, **kwargs)``."""
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = key_fn(*args, **kwargs)
                if self.has(key):
                    return self.get(key)
                result = func(*args, **kwargs)
                self.set(key, result)
                return result
            return wrapper  # type: ignore[return-value]
        return decorator


def hash_dict(d: dict) -> str:
    """Return the first 16 hex chars of the SHA-256 of a JSON-serialized dict."""
    raw = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

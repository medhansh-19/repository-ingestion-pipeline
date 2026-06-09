"""JSON filesystem cache with TTL support."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


class FileCache:
    def __init__(self, root: str | Path = "cache", *, default_ttl_seconds: int = 24 * 60 * 60) -> None:
        self.root = Path(root)
        self.default_ttl_seconds = default_ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, namespace: str, key: str, *, ttl_seconds: int | None = None) -> Any | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return None
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl > 0 and time.time() - float(envelope.get("created_at", 0)) > ttl:
            path.unlink(missing_ok=True)
            return None
        return envelope.get("value")

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"created_at": time.time(), "value": value}, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def get_or_set(self, namespace: str, key: str, loader: Callable[[], Any], *, ttl_seconds: int | None = None) -> Any:
        cached = self.get(namespace, key, ttl_seconds=ttl_seconds)
        if cached is not None:
            return cached
        value = loader()
        self.set(namespace, key, value)
        return value

    def clear(self, namespace: str | None = None) -> None:
        target = self.root if namespace is None else self.root / namespace
        if not target.exists():
            return
        for path in sorted(target.rglob("*.json")):
            path.unlink(missing_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

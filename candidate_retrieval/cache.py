"""Redis-compatible async cache abstractions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any


class InMemoryAsyncCache:
    """Small Redis-like cache useful for tests and local MVP deployments."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        entry = self._values.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if expires_at <= time.monotonic():
            self._values.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        return value

    async def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        self._values[key] = (time.monotonic() + ttl_seconds, value)


class RedisAsyncCache:
    """Thin adapter for redis.asyncio clients.

    Values are JSON encoded so this adapter remains compatible with hosted Redis,
    local Redis, Dragonfly, Valkey, and other Redis protocol implementations.
    """

    def __init__(self, redis_client: Any, *, namespace: str = "candidate_retrieval") -> None:
        self._redis = redis_client
        self._namespace = namespace

    async def get(self, key: str) -> Any | None:
        raw = await self._redis.get(self._cache_key(key))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        await self._redis.setex(
            self._cache_key(key),
            ttl_seconds,
            json.dumps(_json_safe(value), separators=(",", ":")),
        )

    def _cache_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

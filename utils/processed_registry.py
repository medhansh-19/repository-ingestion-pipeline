"""Persistent registry for repositories already processed by Osiris."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProcessedRepositoryRegistry:
    def __init__(self, path: str | Path = "processed_repositories.json") -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._records = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self._records = {}
            return
        if isinstance(raw, list):
            self._records = {str(item.get("repo_id")): item for item in raw if item.get("repo_id")}
        elif isinstance(raw, dict):
            records = raw.get("repositories", raw)
            self._records = {str(key): value for key, value in records.items()}
        else:
            self._records = {}

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"repositories": self._records}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def contains(self, repo_id: str) -> bool:
        return repo_id in self._records

    def add(
        self,
        *,
        repo_id: str,
        novelty_score: float | None,
        category: str | None,
        status: str,
    ) -> None:
        self._records[repo_id] = {
            "repo_id": repo_id,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
            "novelty_score": novelty_score,
            "category": category,
            "status": status,
        }
        self.save()

    def records(self) -> dict[str, dict[str, Any]]:
        return dict(self._records)

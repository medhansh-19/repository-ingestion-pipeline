"""Dependency interfaces for the candidate retrieval layer."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from .models import RepositoryRecord, UserPersona, VectorSearchHit


class UserPersonaStore(Protocol):
    async def get_persona(self, user_id: str) -> UserPersona | None:
        """Return the current persona for a user, or None for cold start users."""


class VectorRepository(Protocol):
    async def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[VectorSearchHit]:
        """Search semantic vectors and return nearest repository hits."""


class MetadataRepository(Protocol):
    async def fetch_repositories_by_ids(self, repo_ids: Sequence[str]) -> dict[str, RepositoryRecord]:
        """Fetch repository metadata keyed by repo_id."""

    async def search_by_categories(self, terms: Sequence[str], *, limit: int) -> list[RepositoryRecord]:
        """Search repository metadata by topics, languages, frameworks, or domains."""

    async def fetch_trending(self, *, limit: int) -> list[RepositoryRecord]:
        """Return globally trending repositories."""

    async def fetch_fresh(self, *, limit: int, max_age_days: int = 30) -> list[RepositoryRecord]:
        """Return fresh or recently indexed repositories."""

    async def fetch_exploration(
        self,
        *,
        excluded_terms: Sequence[str],
        limit: int,
        max_interaction_count: int | None = None,
    ) -> list[RepositoryRecord]:
        """Return hidden-gem repositories outside a user's dominant interests."""


class AsyncCache(Protocol):
    async def get(self, key: str) -> Any | None:
        """Return cached value or None if absent/expired."""

    async def set(self, key: str, value: Any, *, ttl_seconds: int) -> None:
        """Store value for ttl_seconds."""

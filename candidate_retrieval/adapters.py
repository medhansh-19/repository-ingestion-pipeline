"""Production and in-memory adapters for candidate retrieval dependencies."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable, Sequence

from .models import RepositoryRecord, UserPersona, VectorSearchHit


class InMemoryUserPersonaStore:
    def __init__(self, personas: dict[str, UserPersona] | None = None) -> None:
        self.personas = personas or {}

    async def get_persona(self, user_id: str) -> UserPersona | None:
        return self.personas.get(user_id)


class InMemoryVectorRepository:
    def __init__(self, vectors: dict[str, Sequence[float]], metadata: dict[str, dict[str, Any]] | None = None) -> None:
        self.vectors = {repo_id: [float(v) for v in vector] for repo_id, vector in vectors.items()}
        self.metadata = metadata or {}
        self.fail = False

    async def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[VectorSearchHit]:
        if self.fail:
            raise RuntimeError("simulated qdrant failure")
        query = [float(v) for v in embedding]
        hits: list[VectorSearchHit] = []
        for repo_id, vector in self.vectors.items():
            score = _cosine(query, vector)
            if score_threshold is None or score >= score_threshold:
                hits.append(VectorSearchHit(repo_id=repo_id, score=score, metadata=self.metadata.get(repo_id, {})))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]


class InMemoryMetadataRepository:
    def __init__(self, repositories: Iterable[RepositoryRecord]) -> None:
        self.repositories = {repo.repo_id: repo for repo in repositories}

    async def fetch_repositories_by_ids(self, repo_ids: Sequence[str]) -> dict[str, RepositoryRecord]:
        return {repo_id: self.repositories[repo_id] for repo_id in repo_ids if repo_id in self.repositories}

    async def search_by_categories(self, terms: Sequence[str], *, limit: int) -> list[RepositoryRecord]:
        normalized_terms = {term.lower() for term in terms if term}
        scored: list[tuple[float, RepositoryRecord]] = []
        for repo in self.repositories.values():
            repo_terms = {term.lower() for term in repo.topics + repo.languages + repo.domains}
            overlap = len(normalized_terms.intersection(repo_terms))
            if overlap:
                scored.append((overlap + repo.quality_score + repo.activity_score * 0.25, repo))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [repo for _, repo in scored[:limit]]

    async def fetch_trending(self, *, limit: int) -> list[RepositoryRecord]:
        scored = sorted(
            self.repositories.values(),
            key=lambda repo: (
                0.4 * repo.star_velocity
                + 0.3 * repo.fork_velocity
                + 0.2 * repo.contributor_growth
                + 0.1 * repo.activity_growth
            ),
            reverse=True,
        )
        return scored[:limit]

    async def fetch_fresh(self, *, limit: int, max_age_days: int = 30) -> list[RepositoryRecord]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        fresh = [
            repo
            for repo in self.repositories.values()
            if (repo.created_at and _aware(repo.created_at) >= cutoff)
            or (repo.indexed_at and _aware(repo.indexed_at) >= cutoff)
        ]
        fresh.sort(key=lambda repo: (repo.quality_score, repo.activity_score), reverse=True)
        return fresh[:limit]

    async def fetch_exploration(
        self,
        *,
        excluded_terms: Sequence[str],
        limit: int,
        max_interaction_count: int | None = None,
    ) -> list[RepositoryRecord]:
        excluded = {term.lower() for term in excluded_terms if term}
        candidates: list[RepositoryRecord] = []
        for repo in self.repositories.values():
            repo_terms = {term.lower() for term in repo.topics + repo.languages + repo.domains}
            if excluded.intersection(repo_terms):
                continue
            if max_interaction_count is not None and repo.interaction_count > max_interaction_count:
                continue
            candidates.append(repo)
        candidates.sort(
            key=lambda repo: (
                repo.quality_score * 0.55 + repo.novelty_score * 0.25 + repo.activity_score * 0.20,
                -repo.stars,
            ),
            reverse=True,
        )
        return candidates[:limit]


class QdrantVectorRepository:
    """Async wrapper around qdrant-client semantic search."""

    def __init__(self, client: Any, *, collection_name: str = "osiris_research_corpus") -> None:
        self._client = client
        self._collection_name = collection_name

    async def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        score_threshold: float | None = None,
    ) -> list[VectorSearchHit]:
        def _query() -> list[VectorSearchHit]:
            result = self._client.query_points(
                collection_name=self._collection_name,
                query=list(embedding),
                limit=limit,
                with_payload=True,
                score_threshold=score_threshold,
            )
            return [
                VectorSearchHit(
                    repo_id=str((point.payload or {}).get("repo_id", point.id)),
                    score=float(point.score),
                    metadata=dict(point.payload or {}),
                )
                for point in result.points
            ]

        return await asyncio.to_thread(_query)


class AsyncpgMetadataRepository:
    """Postgres metadata adapter using asyncpg connection pools."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def fetch_repositories_by_ids(self, repo_ids: Sequence[str]) -> dict[str, RepositoryRecord]:
        if not repo_ids:
            return {}
        rows = await self._fetch(
            """
            SELECT * FROM repositories
            WHERE repo_id = ANY($1::text[])
            """,
            list(repo_ids),
        )
        return {record.repo_id: record for record in map(_record_from_row, rows)}

    async def search_by_categories(self, terms: Sequence[str], *, limit: int) -> list[RepositoryRecord]:
        normalized_terms = [term.lower() for term in terms if term]
        if not normalized_terms:
            return []
        rows = await self._fetch(
            """
            SELECT *,
                   cardinality(
                     ARRAY(
                       SELECT unnest(coalesce(topics, '{}') || coalesce(languages, '{}') || coalesce(domains, '{}'))
                       INTERSECT
                       SELECT unnest($1::text[])
                     )
                   ) AS match_count
            FROM repositories
            WHERE coalesce(topics, '{}') && $1::text[]
               OR coalesce(languages, '{}') && $1::text[]
               OR coalesce(domains, '{}') && $1::text[]
            ORDER BY match_count DESC, quality_score DESC, activity_score DESC
            LIMIT $2
            """,
            normalized_terms,
            limit,
        )
        return [_record_from_row(row) for row in rows]

    async def fetch_trending(self, *, limit: int) -> list[RepositoryRecord]:
        rows = await self._fetch(
            """
            SELECT *,
                   (0.4 * coalesce(star_velocity, 0)
                    + 0.3 * coalesce(fork_velocity, 0)
                    + 0.2 * coalesce(contributor_growth, 0)
                    + 0.1 * coalesce(activity_growth, 0)) AS trend_score
            FROM repositories
            ORDER BY trend_score DESC, stars DESC
            LIMIT $1
            """,
            limit,
        )
        return [_record_from_row(row) for row in rows]

    async def fetch_fresh(self, *, limit: int, max_age_days: int = 30) -> list[RepositoryRecord]:
        rows = await self._fetch(
            """
            SELECT * FROM repositories
            WHERE coalesce(created_at, indexed_at, updated_at) >= now() - ($1::int * interval '1 day')
               OR coalesce(indexed_at, updated_at, created_at) >= now() - ($1::int * interval '1 day')
            ORDER BY quality_score DESC, activity_score DESC, coalesce(indexed_at, created_at, updated_at) DESC
            LIMIT $2
            """,
            max_age_days,
            limit,
        )
        return [_record_from_row(row) for row in rows]

    async def fetch_exploration(
        self,
        *,
        excluded_terms: Sequence[str],
        limit: int,
        max_interaction_count: int | None = None,
    ) -> list[RepositoryRecord]:
        rows = await self._fetch(
            """
            SELECT * FROM repositories
            WHERE NOT (coalesce(topics, '{}') && $1::text[])
              AND NOT (coalesce(languages, '{}') && $1::text[])
              AND NOT (coalesce(domains, '{}') && $1::text[])
              AND ($2::int IS NULL OR coalesce(interaction_count, 0) <= $2::int)
            ORDER BY (0.55 * coalesce(quality_score, 0)
                      + 0.25 * coalesce(novelty_score, 0)
                      + 0.20 * coalesce(activity_score, 0)) DESC,
                     stars ASC
            LIMIT $3
            """,
            [term.lower() for term in excluded_terms if term],
            max_interaction_count,
            limit,
        )
        return [_record_from_row(row) for row in rows]

    async def _fetch(self, query: str, *args: Any) -> list[Any]:
        async with self._pool.acquire() as connection:
            return await connection.fetch(query, *args)


def _record_from_row(row: Any) -> RepositoryRecord:
    data = dict(row)
    return RepositoryRecord(
        repo_id=str(data.get("repo_id")),
        full_name=data.get("full_name"),
        description=data.get("description"),
        topics=list(data.get("topics") or []),
        languages=list(data.get("languages") or []),
        domains=list(data.get("domains") or data.get("domain") or []),
        stars=int(data.get("stars") or 0),
        forks=int(data.get("forks") or 0),
        pr_count=int(data.get("pr_count") or 0),
        quality_score=float(data.get("quality_score") or 0.0),
        novelty_score=float(data.get("novelty_score") or 0.0),
        activity_score=float(data.get("activity_score") or 0.0),
        star_velocity=float(data.get("star_velocity") or 0.0),
        fork_velocity=float(data.get("fork_velocity") or 0.0),
        contributor_growth=float(data.get("contributor_growth") or 0.0),
        activity_growth=float(data.get("activity_growth") or 0.0),
        interaction_count=int(data.get("interaction_count") or 0),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        indexed_at=data.get("indexed_at"),
        metadata=data.get("metadata") or {},
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value

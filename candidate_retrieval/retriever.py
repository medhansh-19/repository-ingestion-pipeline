"""Production candidate retriever orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Awaitable, Callable, Sequence

from .models import CandidateRepo, RepositoryRecord, RetrievalDiagnostics, UserPersona
from .ports import AsyncCache, MetadataRepository, UserPersonaStore, VectorRepository
from .scoring import category_score, clamp, exploration_score, freshness_score, semantic_score, trend_score

LOGGER = logging.getLogger(__name__)

SEMANTIC_SOURCE = "semantic"
CATEGORY_SOURCE = "category"
TRENDING_SOURCE = "trending"
EXPLORATION_SOURCE = "exploration"
FRESHNESS_SOURCE = "freshness"


class CandidateRetriever:
    """Builds a deduplicated repository candidate pool from independent channels."""

    def __init__(
        self,
        *,
        persona_store: UserPersonaStore,
        vector_repository: VectorRepository,
        metadata_repository: MetadataRepository,
        cache: AsyncCache,
        logger: logging.Logger | None = None,
        cache_ttl_seconds: int = 15 * 60,
        semantic_limit: int = 500,
        category_limit: int = 260,
        trending_limit: int = 180,
        freshness_limit: int = 180,
        min_exploration_limit: int = 100,
        max_exploration_limit: int = 220,
    ) -> None:
        self._persona_store = persona_store
        self._vector_repository = vector_repository
        self._metadata_repository = metadata_repository
        self._cache = cache
        self._logger = logger or LOGGER
        self._cache_ttl_seconds = cache_ttl_seconds
        self._semantic_limit = semantic_limit
        self._category_limit = category_limit
        self._trending_limit = trending_limit
        self._freshness_limit = freshness_limit
        self._min_exploration_limit = min_exploration_limit
        self._max_exploration_limit = max_exploration_limit

    async def retrieve(self, user_id: str, limit: int = 1000) -> list[CandidateRepo]:
        started = time.perf_counter()
        persona = await self._persona_store.get_persona(user_id) or UserPersona()
        exploration_limit = self._exploration_limit(persona, limit)

        channel_calls: list[tuple[str, Callable[[], Awaitable[list[CandidateRepo]]]]] = [
            (SEMANTIC_SOURCE, lambda: self._retrieve_semantic(persona, self._semantic_limit)),
            (CATEGORY_SOURCE, lambda: self._retrieve_category(persona, self._category_limit)),
            (TRENDING_SOURCE, lambda: self._retrieve_trending(self._trending_limit)),
            (EXPLORATION_SOURCE, lambda: self._retrieve_exploration(persona, exploration_limit)),
            (FRESHNESS_SOURCE, lambda: self._retrieve_freshness(self._freshness_limit)),
        ]

        channel_results = await asyncio.gather(
            *(self._run_channel(source, call) for source, call in channel_calls),
            return_exceptions=False,
        )
        candidates = merge_candidates([candidate for result in channel_results for candidate in result])
        candidates.sort(key=lambda candidate: candidate.retrieval_score, reverse=True)
        pool = candidates[:limit]

        elapsed_ms = (time.perf_counter() - started) * 1000
        self._logger.info(
            "candidate_retrieval_complete",
            extra={
                "user_id": user_id,
                "retrieval_time_ms": round(elapsed_ms, 3),
                "candidate_count": len(pool),
                "limit": limit,
            },
        )
        return pool

    async def _run_channel(
        self,
        source: str,
        call: Callable[[], Awaitable[list[CandidateRepo]]],
    ) -> list[CandidateRepo]:
        started = time.perf_counter()
        try:
            candidates = await call()
        except Exception as exc:  # pragma: no cover - exact logging branch covered by fallback tests
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._log_diagnostic(
                RetrievalDiagnostics(
                    source=source,
                    candidate_count=0,
                    elapsed_ms=elapsed_ms,
                    error=str(exc),
                )
            )
            return []
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._log_diagnostic(RetrievalDiagnostics(source=source, candidate_count=len(candidates), elapsed_ms=elapsed_ms))
        return candidates

    async def _retrieve_semantic(self, persona: UserPersona, limit: int) -> list[CandidateRepo]:
        if not persona.embedding:
            return []
        hits = await self._vector_repository.semantic_search(persona.embedding, limit=limit)
        metadata_by_id = await self._metadata_repository.fetch_repositories_by_ids([hit.repo_id for hit in hits])
        candidates: list[CandidateRepo] = []
        for hit in hits:
            score = semantic_score(hit.score)
            repo_metadata = metadata_by_id.get(hit.repo_id).to_metadata() if hit.repo_id in metadata_by_id else {}
            candidates.append(
                CandidateRepo(
                    repo_id=hit.repo_id,
                    retrieval_source=SEMANTIC_SOURCE,
                    retrieval_score=score,
                    semantic_score=score,
                    metadata={**hit.metadata, **repo_metadata},
                )
            )
        return candidates

    async def _retrieve_category(self, persona: UserPersona, limit: int) -> list[CandidateRepo]:
        terms = persona.dominant_terms(limit=10)
        if not terms:
            return []
        cache_key = f"category:v1:{','.join(sorted(term.lower() for term in terms))}:{limit}"
        records, hit = await self._get_or_set_records(
            cache_key,
            lambda: self._metadata_repository.search_by_categories(terms, limit=limit),
        )
        self._log_cache(CATEGORY_SOURCE, hit)
        candidates: list[CandidateRepo] = []
        for repo in records:
            score = category_score(repo, persona)
            candidates.append(
                CandidateRepo(
                    repo_id=repo.repo_id,
                    retrieval_source=CATEGORY_SOURCE,
                    retrieval_score=score,
                    category_score=score,
                    metadata=repo.to_metadata(),
                )
            )
        return candidates

    async def _retrieve_trending(self, limit: int) -> list[CandidateRepo]:
        cache_key = f"trending:v1:{limit}"
        records, hit = await self._get_or_set_records(cache_key, lambda: self._metadata_repository.fetch_trending(limit=limit))
        self._log_cache(TRENDING_SOURCE, hit)
        return [
            CandidateRepo(
                repo_id=repo.repo_id,
                retrieval_source=TRENDING_SOURCE,
                retrieval_score=trend_score(repo),
                trend_score=trend_score(repo),
                metadata=repo.to_metadata(),
            )
            for repo in records
        ]

    async def _retrieve_exploration(self, persona: UserPersona, limit: int) -> list[CandidateRepo]:
        records = await self._metadata_repository.fetch_exploration(
            excluded_terms=persona.excluded_terms_for_exploration(limit=12),
            limit=limit,
            max_interaction_count=10,
        )
        return [
            CandidateRepo(
                repo_id=repo.repo_id,
                retrieval_source=EXPLORATION_SOURCE,
                retrieval_score=exploration_score(repo, persona),
                exploration_score=exploration_score(repo, persona),
                metadata=repo.to_metadata(),
            )
            for repo in records
        ]

    async def _retrieve_freshness(self, limit: int) -> list[CandidateRepo]:
        cache_key = f"freshness:v1:{limit}"
        records, hit = await self._get_or_set_records(
            cache_key,
            lambda: self._metadata_repository.fetch_fresh(limit=limit, max_age_days=30),
        )
        self._log_cache(FRESHNESS_SOURCE, hit)
        return [
            CandidateRepo(
                repo_id=repo.repo_id,
                retrieval_source=FRESHNESS_SOURCE,
                retrieval_score=freshness_score(repo),
                freshness_score=freshness_score(repo),
                metadata=repo.to_metadata(),
            )
            for repo in records
        ]

    async def _get_or_set_records(
        self,
        key: str,
        loader: Callable[[], Awaitable[list[RepositoryRecord]]],
    ) -> tuple[list[RepositoryRecord], bool]:
        cached = await self._cache.get(key)
        if cached is not None:
            return [_repo_from_cache(item) for item in cached], True
        records = await loader()
        await self._cache.set(key, [_repo_to_cache(record) for record in records], ttl_seconds=self._cache_ttl_seconds)
        return records, False

    def _exploration_limit(self, persona: UserPersona, final_limit: int) -> int:
        allocation = clamp(persona.exploration_score, 0.10, 0.20)
        allocated = int(final_limit * allocation)
        return max(self._min_exploration_limit, min(self._max_exploration_limit, allocated))

    def _log_diagnostic(self, diagnostic: RetrievalDiagnostics) -> None:
        self._logger.info(
            "candidate_retrieval_channel",
            extra={
                "retrieval_source": diagnostic.source,
                "candidate_count": diagnostic.candidate_count,
                "retrieval_time_ms": round(diagnostic.elapsed_ms, 3),
                "cache_hits": int(diagnostic.cache_hit),
                "cache_misses": int(diagnostic.cache_miss),
                "error": diagnostic.error,
            },
        )

    def _log_cache(self, source: str, hit: bool) -> None:
        self._logger.info(
            "candidate_retrieval_cache",
            extra={
                "retrieval_source": source,
                "cache_hits": int(hit),
                "cache_misses": int(not hit),
            },
        )


def merge_candidates(candidates: Sequence[CandidateRepo]) -> list[CandidateRepo]:
    merged: dict[str, CandidateRepo] = {}
    source_score_keys = {
        SEMANTIC_SOURCE: "semantic_score",
        CATEGORY_SOURCE: "category_score",
        TRENDING_SOURCE: "trend_score",
        FRESHNESS_SOURCE: "freshness_score",
        EXPLORATION_SOURCE: "exploration_score",
    }

    for candidate in candidates:
        if candidate.repo_id not in merged:
            clone = CandidateRepo(
                repo_id=candidate.repo_id,
                retrieval_source=candidate.retrieval_source,
                retrieval_score=clamp(candidate.retrieval_score),
                semantic_score=candidate.semantic_score,
                category_score=candidate.category_score,
                trend_score=candidate.trend_score,
                freshness_score=candidate.freshness_score,
                exploration_score=candidate.exploration_score,
                metadata=dict(candidate.metadata),
            )
            clone.metadata["retrieval_sources"] = [candidate.retrieval_source]
            clone.metadata["source_scores"] = {candidate.retrieval_source: clamp(candidate.retrieval_score)}
            merged[candidate.repo_id] = clone
            continue

        existing = merged[candidate.repo_id]
        sources = list(existing.metadata.get("retrieval_sources", []))
        if candidate.retrieval_source not in sources:
            sources.append(candidate.retrieval_source)
        existing.metadata["retrieval_sources"] = sources
        source_scores = dict(existing.metadata.get("source_scores", {}))
        source_scores[candidate.retrieval_source] = max(
            float(source_scores.get(candidate.retrieval_source, 0.0)),
            clamp(candidate.retrieval_score),
        )
        existing.metadata["source_scores"] = source_scores
        existing.retrieval_source = ",".join(sources)

        for source, attr in source_score_keys.items():
            new_score = getattr(candidate, attr)
            if new_score is not None:
                old_score = getattr(existing, attr)
                setattr(existing, attr, max(old_score or 0.0, new_score))

        existing.metadata.update({k: v for k, v in candidate.metadata.items() if k not in {"retrieval_sources", "source_scores"}})
        existing.retrieval_score = _aggregate_scores(source_scores.values())

    return list(merged.values())


def _aggregate_scores(scores: Sequence[float]) -> float:
    miss_probability = 1.0
    for score in scores:
        miss_probability *= 1.0 - clamp(float(score))
    return clamp(1.0 - miss_probability)


def _repo_to_cache(repo: RepositoryRecord) -> dict[str, Any]:
    return repo.to_metadata() | {
        "star_velocity": repo.star_velocity,
        "fork_velocity": repo.fork_velocity,
        "contributor_growth": repo.contributor_growth,
        "activity_growth": repo.activity_growth,
        "interaction_count": repo.interaction_count,
    }


def _repo_from_cache(value: Any) -> RepositoryRecord:
    if isinstance(value, RepositoryRecord):
        return value
    data = dict(value)
    return RepositoryRecord(
        repo_id=str(data.get("repo_id")),
        full_name=data.get("full_name"),
        description=data.get("description"),
        topics=list(data.get("topics") or []),
        languages=list(data.get("languages") or []),
        domains=list(data.get("domains") or []),
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
        created_at=_parse_datetime(data.get("created_at")),
        updated_at=_parse_datetime(data.get("updated_at")),
        indexed_at=_parse_datetime(data.get("indexed_at")),
        metadata=data.get("metadata") or {},
    )


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return None

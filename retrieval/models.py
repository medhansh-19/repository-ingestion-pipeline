"""Data models for the repository candidate retrieval layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class CandidateRepo:
    repo_id: str
    retrieval_source: str
    retrieval_score: float
    semantic_score: float | None = None
    category_score: float | None = None
    trend_score: float | None = None
    freshness_score: float | None = None
    exploration_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepositoryRecord:
    repo_id: str
    full_name: str | None = None
    description: str | None = None
    topics: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    stars: int = 0
    forks: int = 0
    pr_count: int = 0
    quality_score: float = 0.0
    novelty_score: float = 0.0
    activity_score: float = 0.0
    star_velocity: float = 0.0
    fork_velocity: float = 0.0
    contributor_growth: float = 0.0
    activity_growth: float = 0.0
    interaction_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    indexed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "full_name": self.full_name,
            "description": self.description,
            "topics": self.topics,
            "languages": self.languages,
            "domains": self.domains,
            "stars": self.stars,
            "forks": self.forks,
            "pr_count": self.pr_count,
            "quality_score": self.quality_score,
            "novelty_score": self.novelty_score,
            "activity_score": self.activity_score,
            "created_at": _iso_or_none(self.created_at),
            "updated_at": _iso_or_none(self.updated_at),
            "indexed_at": _iso_or_none(self.indexed_at),
            **self.metadata,
        }


@dataclass(slots=True)
class UserPersona:
    interest_scores: dict[str, float] = field(default_factory=dict)
    language_scores: dict[str, float] = field(default_factory=dict)
    framework_scores: dict[str, float] = field(default_factory=dict)
    domain_scores: dict[str, float] = field(default_factory=dict)
    novelty_preference: float = 0.5
    exploration_score: float = 0.1
    embedding: list[float] | None = None

    def dominant_terms(self, limit: int = 8) -> list[str]:
        scored_terms: list[tuple[str, float]] = []
        for bucket in (
            self.interest_scores,
            self.language_scores,
            self.framework_scores,
            self.domain_scores,
        ):
            scored_terms.extend((term, score) for term, score in bucket.items() if score > 0)
        scored_terms.sort(key=lambda item: item[1], reverse=True)
        seen: set[str] = set()
        terms: list[str] = []
        for term, _ in scored_terms:
            normalized = term.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                terms.append(term)
            if len(terms) >= limit:
                break
        return terms

    def excluded_terms_for_exploration(self, limit: int = 12) -> list[str]:
        return [term.lower() for term in self.dominant_terms(limit=limit)]


@dataclass(slots=True)
class VectorSearchHit:
    repo_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalDiagnostics:
    source: str
    candidate_count: int
    elapsed_ms: float
    cache_hit: bool = False
    cache_miss: bool = False
    error: str | None = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()

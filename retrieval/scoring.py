"""Scoring helpers for retrieval channels."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from .models import RepositoryRecord, UserPersona


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def trend_score(repo: RepositoryRecord) -> float:
    return clamp(
        0.4 * repo.star_velocity
        + 0.3 * repo.fork_velocity
        + 0.2 * repo.contributor_growth
        + 0.1 * repo.activity_growth
    )


def freshness_score(repo: RepositoryRecord, *, now: datetime | None = None) -> float:
    clock = now or datetime.now(timezone.utc)
    freshness_anchor = repo.indexed_at or repo.created_at or repo.updated_at
    if freshness_anchor is None:
        return 0.0
    if freshness_anchor.tzinfo is None:
        freshness_anchor = freshness_anchor.replace(tzinfo=timezone.utc)
    age_days = max((clock - freshness_anchor).total_seconds() / 86_400, 0.0)
    time_decay = math.exp(-age_days * math.log(2) / 14.0)
    activity_boost = 0.70 + 0.30 * clamp(repo.activity_score)
    return clamp(time_decay * clamp(repo.quality_score) * activity_boost)


def category_score(repo: RepositoryRecord, persona: UserPersona) -> float:
    weighted_terms = _persona_weights(persona)
    if not weighted_terms:
        return 0.0
    repo_terms = {term.lower() for term in repo.topics + repo.languages + repo.domains}
    matched_weight = sum(weight for term, weight in weighted_terms.items() if term in repo_terms)
    max_weight = sum(weighted_terms.values()) or 1.0
    quality_component = 0.25 * clamp(repo.quality_score)
    return clamp((matched_weight / max_weight) * 0.75 + quality_component)


def exploration_score(repo: RepositoryRecord, persona: UserPersona) -> float:
    dominant_terms = {term.lower() for term in persona.dominant_terms(limit=12)}
    repo_terms = {term.lower() for term in repo.topics + repo.languages + repo.domains}
    overlap_penalty = len(dominant_terms.intersection(repo_terms)) / max(len(dominant_terms), 1)
    hidden_gem_score = 0.55 * clamp(repo.quality_score) + 0.25 * clamp(repo.novelty_score) + 0.20 * clamp(repo.activity_score)
    interaction_penalty = min(repo.interaction_count / 25.0, 1.0) * 0.35
    return clamp(hidden_gem_score * (1.0 - overlap_penalty) - interaction_penalty)


def semantic_score(raw_score: float) -> float:
    return clamp(raw_score)


def _persona_weights(persona: UserPersona) -> dict[str, float]:
    merged: dict[str, float] = {}
    for bucket in (
        persona.interest_scores,
        persona.language_scores,
        persona.framework_scores,
        persona.domain_scores,
    ):
        for term, weight in bucket.items():
            normalized = term.strip().lower()
            if normalized:
                merged[normalized] = max(merged.get(normalized, 0.0), clamp(float(weight)))
    return merged

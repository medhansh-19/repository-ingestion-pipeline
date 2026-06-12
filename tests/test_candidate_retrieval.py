from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from retrieval import (
    CandidateRepo,
    CandidateRetriever,
    InMemoryAsyncCache,
    InMemoryMetadataRepository,
    InMemoryUserPersonaStore,
    InMemoryVectorRepository,
    RepositoryRecord,
    UserPersona,
    merge_candidates,
)


def test_ann_semantic_retrieval_returns_qdrant_hits() -> None:
    candidates, cache, _ = asyncio.run(_retrieve(limit=20))

    semantic = [candidate for candidate in candidates if "semantic" in candidate.metadata["retrieval_sources"]]

    assert semantic
    assert semantic[0].semantic_score is not None
    assert cache.misses >= 2


def test_category_retrieval_matches_dominant_persona_terms() -> None:
    candidates, _, _ = asyncio.run(_retrieve(limit=20))

    category_repo = next(candidate for candidate in candidates if candidate.repo_id == "repo-ai-rag")

    assert "category" in category_repo.metadata["retrieval_sources"]
    assert category_repo.category_score is not None
    assert category_repo.category_score > 0.5


def test_trending_retrieval_uses_cached_global_pool() -> None:
    candidates, cache, _ = asyncio.run(_retrieve_twice(limit=20))

    trending = [candidate for candidate in candidates if "trending" in candidate.metadata["retrieval_sources"]]

    assert trending
    assert cache.hits >= 2


def test_exploration_retrieval_avoids_dominant_interests() -> None:
    candidates, _, _ = asyncio.run(_retrieve(limit=30))

    exploration = [candidate for candidate in candidates if "exploration" in candidate.metadata["retrieval_sources"]]

    assert exploration
    assert all("ai" not in [topic.lower() for topic in candidate.metadata.get("topics", [])] for candidate in exploration)


def test_freshness_retrieval_surfaces_recent_repositories() -> None:
    candidates, _, _ = asyncio.run(_retrieve(limit=20))

    fresh = next(candidate for candidate in candidates if candidate.repo_id == "repo-fresh-cli")

    assert "freshness" in fresh.metadata["retrieval_sources"]
    assert fresh.freshness_score is not None
    assert fresh.freshness_score > 0


def test_merge_candidates_deduplicates_and_aggregates_sources() -> None:
    merged = merge_candidates(
        [
            CandidateRepo("repo-1", "semantic", 0.8, semantic_score=0.8, metadata={"name": "Repo"}),
            CandidateRepo("repo-1", "trending", 0.5, trend_score=0.5, metadata={"stars": 10}),
        ]
    )

    assert len(merged) == 1
    assert merged[0].retrieval_source == "semantic,trending"
    assert merged[0].retrieval_score == 0.9
    assert merged[0].metadata["retrieval_sources"] == ["semantic", "trending"]
    assert merged[0].metadata["source_scores"] == {"semantic": 0.8, "trending": 0.5}


def test_empty_user_persona_still_returns_global_and_exploration_candidates() -> None:
    candidates, _, _ = asyncio.run(_retrieve(user_id="cold", persona=UserPersona(), limit=20))

    sources = {source for candidate in candidates for source in candidate.metadata["retrieval_sources"]}

    assert "semantic" not in sources
    assert "category" not in sources
    assert "trending" in sources
    assert "freshness" in sources
    assert "exploration" in sources


def test_cold_start_user_without_persona_falls_back_to_global_pools() -> None:
    repos = _repositories()
    retriever = CandidateRetriever(
        persona_store=InMemoryUserPersonaStore({}),
        vector_repository=InMemoryVectorRepository(_vectors()),
        metadata_repository=InMemoryMetadataRepository(repos),
        cache=InMemoryAsyncCache(),
    )

    candidates = asyncio.run(retriever.retrieve("missing-user", limit=20))
    sources = {source for candidate in candidates for source in candidate.metadata["retrieval_sources"]}

    assert "trending" in sources
    assert "freshness" in sources


def test_missing_embedding_skips_ann_without_failing() -> None:
    persona = UserPersona(interest_scores={"AI": 1.0}, embedding=None)
    candidates, _, _ = asyncio.run(_retrieve(persona=persona, limit=20))

    assert candidates
    assert all(candidate.semantic_score is None for candidate in candidates if candidate.repo_id != "repo-ai-rag")


def test_qdrant_failure_falls_back_to_other_channels() -> None:
    candidates, _, vector_repo = asyncio.run(_retrieve(vector_fail=True, limit=20))

    sources = {source for candidate in candidates for source in candidate.metadata["retrieval_sources"]}

    assert vector_repo.fail is True
    assert "semantic" not in sources
    assert "trending" in sources
    assert "freshness" in sources


def test_integration_candidate_pool_contains_expected_shape() -> None:
    candidates, _, _ = asyncio.run(_retrieve(limit=5))

    assert len(candidates) <= 5
    assert candidates[0].repo_id
    assert isinstance(candidates[0].metadata, dict)
    assert "retrieval_sources" in candidates[0].metadata
    assert "source_scores" in candidates[0].metadata


async def _retrieve(
    *,
    user_id: str = "user-1",
    persona: UserPersona | None = None,
    vector_fail: bool = False,
    limit: int = 1000,
) -> tuple[list[CandidateRepo], InMemoryAsyncCache, InMemoryVectorRepository]:
    repos = _repositories()
    cache = InMemoryAsyncCache()
    vector_repo = InMemoryVectorRepository(_vectors())
    vector_repo.fail = vector_fail
    retriever = CandidateRetriever(
        persona_store=InMemoryUserPersonaStore({user_id: persona or _persona()}),
        vector_repository=vector_repo,
        metadata_repository=InMemoryMetadataRepository(repos),
        cache=cache,
        semantic_limit=5,
        category_limit=5,
        trending_limit=5,
        freshness_limit=5,
        min_exploration_limit=2,
        max_exploration_limit=5,
    )
    return await retriever.retrieve(user_id, limit=limit), cache, vector_repo


async def _retrieve_twice(*, limit: int) -> tuple[list[CandidateRepo], InMemoryAsyncCache, InMemoryVectorRepository]:
    repos = _repositories()
    cache = InMemoryAsyncCache()
    vector_repo = InMemoryVectorRepository(_vectors())
    retriever = CandidateRetriever(
        persona_store=InMemoryUserPersonaStore({"user-1": _persona()}),
        vector_repository=vector_repo,
        metadata_repository=InMemoryMetadataRepository(repos),
        cache=cache,
        semantic_limit=5,
        category_limit=5,
        trending_limit=5,
        freshness_limit=5,
        min_exploration_limit=2,
        max_exploration_limit=5,
    )
    await retriever.retrieve("user-1", limit=limit)
    second = await retriever.retrieve("user-1", limit=limit)
    return second, cache, vector_repo


def _persona() -> UserPersona:
    return UserPersona(
        interest_scores={"AI": 0.95, "RAG": 0.88, "Agents": 0.82},
        language_scores={"Python": 0.7},
        domain_scores={"developer-tools": 0.6},
        novelty_preference=0.7,
        exploration_score=0.15,
        embedding=[1.0, 0.0, 0.0],
    )


def _vectors() -> dict[str, list[float]]:
    return {
        "repo-ai-rag": [0.98, 0.1, 0.0],
        "repo-agent-framework": [0.92, 0.15, 0.0],
        "repo-rust-db": [0.0, 1.0, 0.0],
        "repo-fresh-cli": [0.2, 0.8, 0.1],
        "repo-design-system": [0.0, 0.1, 0.9],
    }


def _repositories() -> list[RepositoryRecord]:
    now = datetime.now(timezone.utc)
    return [
        RepositoryRecord(
            repo_id="repo-ai-rag",
            full_name="acme/ai-rag",
            topics=["AI", "RAG"],
            languages=["Python"],
            domains=["developer-tools"],
            stars=900,
            forks=80,
            quality_score=0.95,
            novelty_score=0.75,
            activity_score=0.9,
            star_velocity=0.9,
            fork_velocity=0.6,
            contributor_growth=0.7,
            activity_growth=0.8,
            created_at=now - timedelta(days=100),
            updated_at=now - timedelta(days=1),
            indexed_at=now - timedelta(days=80),
        ),
        RepositoryRecord(
            repo_id="repo-agent-framework",
            full_name="acme/agent-framework",
            topics=["Agents", "LLM"],
            languages=["Python"],
            domains=["developer-tools"],
            stars=600,
            forks=40,
            quality_score=0.88,
            novelty_score=0.7,
            activity_score=0.85,
            star_velocity=0.7,
            fork_velocity=0.5,
            contributor_growth=0.5,
            activity_growth=0.7,
            created_at=now - timedelta(days=60),
            updated_at=now - timedelta(days=3),
            indexed_at=now - timedelta(days=50),
        ),
        RepositoryRecord(
            repo_id="repo-rust-db",
            full_name="acme/rust-db",
            topics=["Database", "Storage"],
            languages=["Rust"],
            domains=["systems"],
            stars=120,
            forks=12,
            quality_score=0.82,
            novelty_score=0.92,
            activity_score=0.7,
            star_velocity=0.4,
            fork_velocity=0.3,
            contributor_growth=0.6,
            activity_growth=0.4,
            created_at=now - timedelta(days=200),
            updated_at=now - timedelta(days=7),
            indexed_at=now - timedelta(days=190),
        ),
        RepositoryRecord(
            repo_id="repo-fresh-cli",
            full_name="acme/fresh-cli",
            topics=["CLI", "Automation"],
            languages=["Go"],
            domains=["devops"],
            stars=80,
            forks=7,
            quality_score=0.8,
            novelty_score=0.8,
            activity_score=0.95,
            star_velocity=0.8,
            fork_velocity=0.6,
            contributor_growth=0.5,
            activity_growth=0.9,
            created_at=now - timedelta(days=5),
            updated_at=now - timedelta(days=1),
            indexed_at=now - timedelta(days=1),
        ),
        RepositoryRecord(
            repo_id="repo-design-system",
            full_name="acme/design-system",
            topics=["Design", "Components"],
            languages=["TypeScript"],
            domains=["frontend"],
            stars=150,
            forks=18,
            quality_score=0.84,
            novelty_score=0.86,
            activity_score=0.72,
            star_velocity=0.35,
            fork_velocity=0.2,
            contributor_growth=0.3,
            activity_growth=0.4,
            interaction_count=3,
            created_at=now - timedelta(days=90),
            updated_at=now - timedelta(days=2),
            indexed_at=now - timedelta(days=85),
        ),
    ]

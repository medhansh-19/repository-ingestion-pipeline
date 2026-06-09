# Candidate Retrieval Layer

This package implements the MVP candidate retrieval layer for the GitHub repository recommendation engine.

## Pipeline Position

```text
Repository Ingestion Layer
-> Embedding Pipeline
-> Storage Layer
-> Candidate Retrieval Layer
-> Heavy Filtering Layer
-> Ranking Layer
-> Feed Assembly
```

The layer returns a broad, deduplicated pool of repository candidates before filtering and ranking. It does not make final feed-ordering decisions.

## Public Interface

```python
from candidate_retrieval import CandidateRetriever

candidates = await retriever.retrieve(user_id="user-123", limit=1000)
```

Each result is a `CandidateRepo` with per-channel scores, source attribution, and repository metadata.

## Retrieval Channels

The retriever runs all channels concurrently with `asyncio.gather`:

1. `semantic`: Qdrant ANN search over repository embeddings using the current user persona embedding.
2. `category`: Postgres metadata search over dominant persona topics, languages, frameworks, and domains.
3. `trending`: Cached global pool scored by star, fork, contributor, and activity velocity.
4. `exploration`: User-specific hidden-gem pool outside dominant interests.
5. `freshness`: Cached global pool for newly created or recently indexed repositories.

Trending and freshness are intentionally global cached pools for MVP latency. They do not depend on `user_embedding`.

## Storage Dependencies

The core retriever depends on ports, not concrete infrastructure:

- `UserPersonaStore`: loads the session-updated `UserPersona`.
- `VectorRepository`: performs ANN semantic search.
- `MetadataRepository`: reads repository metadata from Postgres or an equivalent store.
- `AsyncCache`: Redis-compatible cache interface.

Production adapters are included:

- `QdrantVectorRepository`
- `AsyncpgMetadataRepository`
- `RedisAsyncCache`

Test/local adapters are included:

- `InMemoryUserPersonaStore`
- `InMemoryVectorRepository`
- `InMemoryMetadataRepository`
- `InMemoryAsyncCache`

## Postgres Expectations

`AsyncpgMetadataRepository` expects a `repositories` table with these fields where available:

```text
repo_id text primary key
full_name text
description text
topics text[]
languages text[]
domains text[]
stars int
forks int
pr_count int
quality_score double precision
novelty_score double precision
activity_score double precision
star_velocity double precision
fork_velocity double precision
contributor_growth double precision
activity_growth double precision
interaction_count int
created_at timestamptz
updated_at timestamptz
indexed_at timestamptz
metadata jsonb
```

If the current schema has differently named velocity fields, map them in a view or adjust `AsyncpgMetadataRepository`.

## Qdrant Expectations

`QdrantVectorRepository` expects points in collection `osiris_research_corpus` by default, with vectors stored as repository embeddings and payloads containing:

```json
{
  "repo_id": "owner/repo"
}
```

Additional payload metadata is passed through and merged with Postgres metadata when available.

## Caching

The Redis-compatible cache stores:

- Category candidate pools by dominant persona term set.
- Trending global pools by limit.
- Freshness global pools by limit.

Default TTL is 15 minutes.

## Merge Semantics

`merge_candidates()` deduplicates by `repo_id`, preserves all channels, stores channel scores in `metadata["source_scores"]`, stores source names in `metadata["retrieval_sources"]`, and aggregates scores using a probabilistic union:

```text
1 - product(1 - source_score)
```

This rewards repositories surfaced by multiple independent channels without requiring all score types to share identical distributions.

## Failure Handling

A channel failure logs diagnostics and returns an empty list for that channel. This allows Qdrant outages to degrade gracefully to category, trending, exploration, and freshness candidates.

## Performance Notes

- All retrieval channels are async and run concurrently.
- Trending and freshness avoid per-user query work via cached global pools.
- Category results are cached by dominant term set.
- The default channel limits intentionally over-fetch before deduplication to target approximately 1000 candidates.

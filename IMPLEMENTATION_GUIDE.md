# Osiris GitHub Acquisition Implementation Guide

This guide explains how to deploy, configure, operate, debug, and extend the GitHub-powered acquisition layer for the Osiris repository ingestion pipeline.

## System Overview

Osiris already contains the analytical core in `ingestion_engine.py`. The acquisition layer added in this implementation does not replace that engine. It feeds live GitHub repositories into the existing pipeline.

```text
GitHub Discovery
-> GitHub Fetch
-> Repository Enrichment
-> README Processing
-> Osiris Payload Conversion
-> Embedding Generation
-> Qdrant Indexing
-> Novelty Analysis
-> Taxonomy Classification
-> Reporting
```

The main production entrypoint is:

```bash
python3 github_ingestion_runner.py
```

The runner discovers repositories, enriches them, converts them into Osiris-compatible dictionaries, and calls the existing `ingest_batch()` function.

## Data Flow

1. `github_discovery.py` creates balanced GitHub Search API queries across technology categories and repository maturity bands.
2. `github_client.py` executes authenticated GitHub REST API requests with pagination, retries, rate-limit handling, and timeouts.
3. `repository_enricher.py` fetches full metadata, README, topics, languages, contributors, events, and recent stargazer timestamps.
4. `readme_processor.py` decodes README content, cleans markdown, removes badges/images/boilerplate, and extracts meaningful paragraphs.
5. `repository_enricher.py` emits Osiris payloads with fields expected by `ingestion_engine.py`.
6. `github_ingestion_runner.py` passes payloads into `ingest_batch()`.
7. `ingestion_engine.py` runs documentation scoring, activity scoring, trend velocity scoring, taxonomy classification, hybrid embedding generation, Qdrant neighbor search, novelty computation, Qdrant insertion, and reporting.
8. `processed_repositories.json` records processed repositories and prevents duplicate ingestion.

## File Structure

`github_client.py`

GitHub REST API client. Handles `GITHUB_TOKEN`, request retries, pagination, timeouts, rate-limit waits, and graceful optional-resource failures.

`github_discovery.py`

Balanced discovery engine. It searches across AI, LLM, RAG, agents, frontend, backend, security, infrastructure, databases, observability, DevOps, cloud, automation, ML, robotics, systems, mobile, game development, and other categories.

`repository_enricher.py`

Transforms live GitHub API responses into Osiris-compatible payloads. It estimates fields GitHub does not provide directly, such as star velocity and README-to-codebase ratio.

`readme_processor.py`

Decodes GitHub README payloads and extracts high-signal paragraphs for `extract_tags()`, `classify_category()`, and `generate_hybrid_embedding()` inside `ingestion_engine.py`.

`github_ingestion_runner.py`

Primary acquisition entrypoint. It orchestrates discovery, cache lookups, enrichment, duplicate skipping, ingestion, report printing, and processed registry updates.

`cache/file_cache.py`

TTL-aware JSON filesystem cache used for discovery results, raw repository responses, README markdown, and processed Osiris payloads.

`cache/processed_registry.py`

Persistent registry for repositories already evaluated by Osiris.

`processed_repositories.json`

Default corpus registry file. It stores `repo_id`, `ingestion_timestamp`, `novelty_score`, `category`, and `status`.

`IMPLEMENTATION_GUIDE.md`

This guide.

`tests/test_github_acquisition.py`

Unit and regression tests for README processing, enrichment, discovery, and novelty neighbor fallback behavior.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

If you use the existing repository venv, run:

```bash
.venv/bin/python -m pytest -q
```

## Environment Variables

`GITHUB_TOKEN`

GitHub personal access token. Strongly recommended because unauthenticated GitHub Search API limits are low.

`QDRANT_URL`

Optional Qdrant endpoint. If unset, `ingestion_engine.py` uses an in-memory Qdrant client when `qdrant-client` is available. For persistent corpus growth, set this to a real Qdrant instance.

Example:

```bash
export QDRANT_URL="http://localhost:6333"
```

`QDRANT_API_KEY`

Optional Qdrant API key for Qdrant Cloud or protected deployments.

`OSIRIS_DISCOVER_LIMIT`

Optional default number of repositories to discover when running `github_ingestion_runner.py`.

`OSIRIS_INGEST_LIMIT`

Optional default number of enriched repositories to ingest in one run.

`OPENAI_API_KEY`

Not required by the current code path. Documented here because future embedding or summarization extensions may use it. The current engine uses `sentence-transformers` when installed and a deterministic fallback embedding otherwise.

## Initial Setup

1. Create a GitHub personal access token.
2. Export the token:

```bash
export GITHUB_TOKEN="github_pat_..."
```

3. Start Qdrant locally for persistent vector storage:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

4. Point Osiris at Qdrant:

```bash
export QDRANT_URL="http://localhost:6333"
```

5. Verify tests pass:

```bash
.venv/bin/python -m pytest -q
```

6. Run a dry-run ingestion preview:

```bash
.venv/bin/python github_ingestion_runner.py --dry-run --discover-limit 20 --ingest-limit 5
```

7. Run live ingestion:

```bash
.venv/bin/python github_ingestion_runner.py --discover-limit 80 --ingest-limit 25
```

## Running The System

Run live GitHub acquisition and ingestion:

```bash
python3 github_ingestion_runner.py
```

Run with smaller limits while debugging:

```bash
python3 github_ingestion_runner.py --discover-limit 20 --ingest-limit 5
```

Preview enriched payloads without calling `ingest_batch()`:

```bash
python3 github_ingestion_runner.py --dry-run --preview-count 3
```

Force fresh discovery queries:

```bash
python3 github_ingestion_runner.py --refresh-discovery
```

Force refetching repository metadata and README content:

```bash
python3 github_ingestion_runner.py --refresh-repositories
```

Allow already processed repositories to be evaluated again:

```bash
python3 github_ingestion_runner.py --reprocess
```

Run the original staged JSON workflow:

```bash
python3 ingestion_engine.py
```

That command preserves the old behavior. It reads `staged_repositories.json` if present and runs the analytical engine directly.

## Cache System

The acquisition layer uses filesystem JSON caches under `cache/`.

Namespaces:

`cache/discovery/`

Stores GitHub discovery search results.

`cache/raw_repositories/`

Stores raw GitHub repository API responses.

`cache/readmes/`

Stores raw README markdown.

`cache/processed/`

Stores Osiris-compatible enriched payloads.

Default expiration:

- Discovery cache: 6 hours by runner default.
- Processed payload cache: 7 days.
- FileCache default TTL: 24 hours unless overridden.

Clear all cached JSON files:

```bash
find cache -name '*.json' -delete
```

Clear one namespace:

```bash
rm -f cache/discovery/*.json
```

The cache package itself lives in `cache/*.py`; only JSON cache files should be deleted.

## GitHub Discovery Strategy

Discovery is intentionally balanced. It is not a star-only scraper.

Technology categories include:

- AI
- LLM
- RAG
- AI Agents
- Frontend
- Backend
- Developer Tools
- Web Frameworks
- Security
- Infrastructure
- Databases
- Observability
- DevOps
- Cloud
- Automation
- ML
- Computer Vision
- Robotics
- Bioinformatics
- Embedded Systems
- Systems Programming
- Mobile
- Game Development

Repository maturity mix:

- 40% high-star projects: established ecosystem anchors.
- 30% recently active projects: currently maintained repositories.
- 20% mid-sized projects: useful but not necessarily famous projects.
- 10% emerging projects: newer hidden-gem candidates.

Deduplication is by `full_name` before enrichment. Each repository carries `_discovery_category` and `_discovery_band` into the enriched Osiris payload.

## Repository Enrichment

`RepositoryEnricher` generates these Osiris fields:

`id`

GitHub `full_name`, such as `owner/repository`.

`star_count`

GitHub `stargazers_count`.

`primary_language`

GitHub `language`, falling back to the largest entry from the languages endpoint.

`mentionable_users_count`

Contributor count from `/contributors`, capped at 100. If contributors are unavailable, falls back to 1 when an owner exists.

`readme_length`

Raw README markdown character count.

`readme_to_codebase_ratio`

`readme_length / max(repository_size_kb * 1024, 1)`.

`delta_3d`, `delta_7d`, `delta_30d`

Estimated star velocity. Preferred source is the stargazers endpoint with `application/vnd.github.star+json`, which includes `starred_at`. If timestamped stargazers are unavailable, the enricher estimates velocity from recent repository events, push recency, and total stars.

`extracted_paragraphs`

Cleaned README paragraphs from `readme_processor.py`.

`pushed_days_ago`

Days since GitHub `pushed_at`.

`topics`

GitHub topics from repository metadata or `/topics`.

`languages`

Language names from `/languages`.

## README Processing

The README processor:

- Decodes Base64 GitHub README payloads.
- Removes shields/badges.
- Removes image links.
- Removes fenced code blocks.
- Removes markdown tables.
- Converts markdown links to plain text.
- Removes HTML tags.
- Removes low-signal boilerplate.
- Keeps meaningful paragraphs with enough natural language content.

This directly improves Osiris stages that depend on textual structure:

- `extract_tags()`
- `classify_category()`
- `build_structured_summary()`
- `generate_hybrid_embedding()`

## Qdrant Integration

Default collection:

```text
osiris_research_corpus
```

Default vector name:

```text
repo_embedding
```

Embedding dimension:

```text
384
```

Payload schema inserted by `ingestion_engine.py`:

```json
{
  "repo_id": "owner/repo",
  "category": "AI Agent",
  "activity": 0.82,
  "tags": ["agent", "llm"],
  "doc_quality": 0.75,
  "trend": 0.42,
  "quadrant": "Viral Rockets"
}
```

The engine now supports both named-vector and unnamed-vector Qdrant collections. New collections are created with the named vector `repo_embedding`, matching the storage architecture.

Insert workflow:

1. Generate hybrid embedding.
2. Compute novelty against existing neighbors.
3. If approved, append to internal corpus history.
4. Upsert Qdrant point with deterministic UUID from `repo_id`.
5. Use `wait=True` so subsequent searches can see the point.
6. Print inserted point ID and collection size.

Search workflow:

1. Verify collection exists.
2. Print collection diagnostics.
3. Count points exactly.
4. Query Qdrant with named or unnamed vector format as appropriate.
5. Print neighbor count, neighbor IDs, and similarity scores.
6. If Qdrant returns no neighbors but the in-process corpus has vectors, fall back to an internal cosine scan.

## Novelty Engine

Novelty is calculated by `compute_multi_dimensional_novelty()` in `ingestion_engine.py`.

Core inputs:

- Target repository category.
- Target repository activity score.
- Extracted tags.
- Top Qdrant nearest neighbors.
- Corpus size.

Dimensions:

- 60% semantic novelty: `1 - mean_neighbor_similarity`.
- 20% tech-stack novelty: Jaccard distance between target tags and nearest neighbor tags.
- 10% category novelty: saturation of matching categories in nearest-neighbor slice.
- 10% activity novelty: target activity relative to neighbor activity.

Thresholds:

- `NOVELTY_THRESHOLD = 0.35`
- `VELOCITY_THRESHOLD = 0.40`
- `DUPLICATE_SIMILARITY_THRESHOLD = 0.94`
- `WRAPPER_SIMILARITY_THRESHOLD = 0.85`

Quadrants:

- Viral Rockets: high novelty, high velocity.
- Hidden Gems: high novelty, lower velocity.
- Copycats / Clones: low novelty, high velocity.
- Dormant Ecosystem Nodes: low novelty, low velocity.

## Diagnostics And Troubleshooting

### GitHub API Rate Limits

Symptoms:

- Runner pauses for a long time.
- Logs mention rate limit exceeded.
- GitHub returns HTTP 403.

Cause:

- Missing token or too many search/enrichment requests.

Resolution:

```bash
export GITHUB_TOKEN="github_pat_..."
python3 github_ingestion_runner.py --discover-limit 40 --ingest-limit 10
```

Use smaller limits while developing.

### Empty README

Symptoms:

- `readme_length` is 0.
- `extracted_paragraphs` is empty.
- Documentation score is low.

Cause:

- Repository has no README.
- README endpoint returned 404.
- README is mostly images, badges, code, or tables.

Resolution:

- Confirm the repository has a README in GitHub.
- Run with `--refresh-repositories` to bypass cached payloads.
- Inspect `cache/readmes/` for raw content.

### Qdrant Returns No Neighbors

Symptoms:

```text
Qdrant corpus_total: 96
Neighbors retrieved: 0
```

Cause:

- Named-vector mismatch.
- Vector dimension mismatch.
- Collection was created with a different schema.
- Query API version mismatch.
- Points exist without vectors or without expected payload.

Resolution:

- Check diagnostics printed by `ingestion_engine.py`:

```text
[DEBUG] Qdrant diagnostics: {...}
[DEBUG] Neighbor IDs: [...]
[DEBUG] Sim Scores: [...]
```

- Confirm `vector_names` contains `repo_embedding` for named-vector collections.
- Confirm `vector_size` equals `384`.
- If the collection was created incorrectly during experimentation, recreate it:

```bash
curl -X DELETE http://localhost:6333/collections/osiris_research_corpus
```

Then rerun ingestion.

### Novelty Always Equals 1.0

Symptoms:

- Every repository is approved as if it is the first seed.
- Reports show no nearest neighbors.

Cause:

- Qdrant returned no neighbors and internal corpus was empty.
- Running every ingestion in a fresh process without persistent Qdrant.
- Collection vector schema mismatch.

Resolution:

- Use persistent Qdrant via `QDRANT_URL`.
- Ensure Qdrant diagnostics show a nonzero point count and nonzero neighbors.
- Run at least two similar repositories in the same batch to validate neighbor behavior.
- Run tests:

```bash
.venv/bin/python -m pytest tests/test_github_acquisition.py -q
```

The test `test_ingestion_engine_uses_internal_neighbors_when_qdrant_unavailable` verifies novelty drops below 1.0 for a similar second repository even without Qdrant.

### Embedding Dimension Mismatch

Symptoms:

- Qdrant warnings mention collection/query dimension mismatch.
- Inserts or searches fail.

Cause:

- Existing collection was created with a vector size other than 384.
- Embedding model changed.

Resolution:

- Keep `EMBEDDING_DIM` and `_FALLBACK_DIM` aligned.
- Recreate the collection if you intentionally change embedding models.

### Authentication Errors

Symptoms:

- GitHub returns HTTP 401.
- Qdrant Cloud returns unauthorized.

Resolution:

```bash
export GITHUB_TOKEN="github_pat_..."
export QDRANT_API_KEY="..."
export QDRANT_URL="https://...qdrant.cloud"
```

Verify the token is available:

```bash
python3 -c 'import os; print(bool(os.getenv("GITHUB_TOKEN")))'
```

## Validation Checklist

Use this checklist before calling the system production-ready:

- GitHub token loaded.
- Discovery returns repositories.
- Discovery includes multiple categories and bands.
- README extraction returns meaningful paragraphs.
- Enriched payloads include required Osiris fields.
- `processed_repositories.json` prevents duplicate ingestion.
- Embeddings generate without errors.
- Qdrant collection exists.
- Qdrant vector dimension is 384.
- Qdrant point count increases after approved ingestion.
- Neighbor retrieval returns IDs and similarity scores.
- Novelty scores vary across repositories.
- Taxonomy classification assigns useful categories.
- Reports are generated.
- Tests pass.

## Future Expansion

Recommended roadmap:

- Multi-source ingestion from GitLab.
- Bitbucket repository acquisition.
- Package registry discovery from PyPI, npm, crates.io, and Maven.
- Repository dependency graphing.
- Architecture extraction from source trees.
- README diagram and flowchart generation.
- Contributor network analysis.
- Longitudinal ecosystem intelligence dashboards.
- Scheduled ingestion jobs with backpressure and rate-limit budgets.
- Persistent Postgres storage for all enriched repository payloads.
- Heavy filtering and ranking layers on top of the candidate retrieval package.

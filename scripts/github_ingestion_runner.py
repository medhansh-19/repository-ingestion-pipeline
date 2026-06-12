"""Live GitHub acquisition entrypoint for the Osiris ingestion engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cache import FileCache, ProcessedRepositoryRegistry
from github_client import GitHubClient
from github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from ingestion_engine import ingest_batch, print_batch_summary
from repository_enricher import RepositoryEnricher


RAW_CACHE_TTL_SECONDS = 6 * 60 * 60
PROCESSED_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60


def main() -> None:
    args = parse_args()
    client = GitHubClient(timeout_seconds=args.timeout, max_retries=args.retries)
    cache = FileCache(args.cache_dir, default_ttl_seconds=PROCESSED_CACHE_TTL_SECONDS)
    registry = ProcessedRepositoryRegistry(args.processed_registry)
    discovery = GitHubDiscoveryEngine(
        client,
        config=DiscoveryConfig(
            total_limit=args.discover_limit,
            per_query=args.per_query,
            pages_per_query=args.pages_per_query,
            random_seed=args.seed,
        ),
    )
    enricher = RepositoryEnricher(client)

    print(f"[OSIRIS] Discovering up to {args.discover_limit} repositories from GitHub...")
    discovered = cache.get("discovery", _discovery_key(args), ttl_seconds=args.discovery_cache_ttl)
    if discovered is None or args.refresh_discovery:
        discovered = discovery.discover(limit=args.discover_limit)
        cache.set("discovery", _discovery_key(args), discovered)
    print(f"[OSIRIS] Discovery returned {len(discovered)} unique repositories.")

    payloads: list[dict[str, Any]] = []
    skipped_processed = 0
    for repo in discovered:
        full_name = repo.get("full_name")
        if not full_name:
            continue
        if registry.contains(full_name) and not args.reprocess:
            skipped_processed += 1
            continue
        if len(payloads) >= args.ingest_limit:
            break

        cached_payload = cache.get("processed", full_name, ttl_seconds=PROCESSED_CACHE_TTL_SECONDS)
        if cached_payload is not None and not args.refresh_repositories:
            payloads.append(cached_payload)
            continue

        print(f"[OSIRIS] Enriching {full_name}...")
        try:
            result = enricher.enrich(repo)
        except Exception as exc:
            print(f"[WARN] Enrichment failed for {full_name}: {exc}")
            continue
        if result is None:
            continue
        cache.set("raw_repositories", full_name, result.raw_repository)
        cache.set("readmes", full_name, result.readme.raw_markdown)
        cache.set("processed", full_name, result.payload)
        payloads.append(result.payload)

    print(f"[OSIRIS] Prepared {len(payloads)} payloads. Skipped {skipped_processed} already-processed repositories.")
    if args.dry_run:
        print(json.dumps(payloads[: args.preview_count], indent=2))
        return
    if not payloads:
        print("[OSIRIS] No new repositories to ingest.")
        return

    results = ingest_batch(payloads, verbose=not args.quiet)
    print_batch_summary(results)

    for result in results:
        registry.add(
            repo_id=result.repo_id,
            novelty_score=result.novelty.final,
            category=result.category,
            status=result.decision,
        )

    print(f"[OSIRIS] Registry updated: {args.processed_registry}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover, enrich, and ingest live GitHub repositories into Osiris.")
    parser.add_argument("--discover-limit", type=int, default=int(os.getenv("OSIRIS_DISCOVER_LIMIT", "120")))
    parser.add_argument("--ingest-limit", type=int, default=int(os.getenv("OSIRIS_INGEST_LIMIT", "40")))
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--pages-per-query", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--processed-registry", default="processed_repositories.json")
    parser.add_argument("--discovery-cache-ttl", type=int, default=RAW_CACHE_TTL_SECONDS)
    parser.add_argument("--refresh-discovery", action="store_true")
    parser.add_argument("--refresh-repositories", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _discovery_key(args: argparse.Namespace) -> str:
    return f"limit={args.discover_limit}:per={args.per_query}:pages={args.pages_per_query}:seed={args.seed}"


if __name__ == "__main__":
    main()

import asyncio
import os
from datetime import datetime, timezone

from github_client import GitHubClient
from github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from repository_enricher import RepositoryEnricher
from cache import FileCache
from ingestion_engine import ingest_batch, print_batch_summary

from candidate_retrieval.adapters import (
    InMemoryUserPersonaStore,
    InMemoryVectorRepository,
    InMemoryMetadataRepository,
)
from candidate_retrieval.cache import InMemoryAsyncCache
from candidate_retrieval.models import RepositoryRecord, UserPersona
from candidate_retrieval.retriever import CandidateRetriever


async def run_end_to_end_test():
    """
    Runs an end-to-end demonstration of the three architectural layers:
    1. Fetching (Discovery + Client + Enricher)
    2. Ingestion (Analysis + Feature extraction)
    3. Candidate Retrieval (Recommender system)
    """
    
    # ---------------------------------------------------------
    # LAYER 1: FETCHING
    # ---------------------------------------------------------
    print("=== [1] FETCHING LAYER ===")
    cache = FileCache("cache")
    client = GitHubClient(timeout_seconds=20.0, max_retries=3)
    discovery = GitHubDiscoveryEngine(
        client,
        config=DiscoveryConfig(total_limit=50, per_query=20, pages_per_query=3)
    )
    
    print("Discovering repositories from GitHub...")
    # Cache discovery results for 6 hours (matching runner default)
    discovery_key = "limit=50:per=20:pages=3"
    
    if os.environ.get("GITHUB_TOKEN"):
        print("Using REAL-TIME GitHub API...")
        discovered = discovery.discover(limit=50)
        cache.set("discovery", discovery_key, discovered)
    else:
        discovered = cache.get("discovery", discovery_key, ttl_seconds=6 * 60 * 60)
        if discovered is None:
            discovered = discovery.discover(limit=50)
            cache.set("discovery", discovery_key, discovered)
            
    print(f"Discovered {len(discovered)} repositories.")

    print("\nEnriching payloads with full readmes and commit histories...")
    enricher = RepositoryEnricher(client)
    payloads = []
    for repo in discovered:
        full_name = repo.get("full_name")
        if not full_name:
            continue
        # Try retrieving from cache first, unless we want real-time
        if not os.environ.get("GITHUB_TOKEN"):
            cached_payload = cache.get("processed", full_name, ttl_seconds=7 * 24 * 60 * 60)
            if cached_payload is not None:
                payloads.append(cached_payload)
                continue
                
        print(f" - Fetching data for {full_name}...")
        try:
            result = enricher.enrich(repo)
            if result:
                cache.set("raw_repositories", full_name, result.raw_repository)
                cache.set("readmes", full_name, result.readme.raw_markdown)
                cache.set("processed", full_name, result.payload)
                payloads.append(result.payload)
        except Exception as e:
            print(f"Failed to enrich {full_name}: {e}")

    if not payloads:
        print("No payloads enriched. Cannot proceed to ingestion.")
        return

    # ---------------------------------------------------------
    # LAYER 2: INGESTION
    # ---------------------------------------------------------
    print("\n=== [2] INGESTION LAYER ===")
    print(f"Running Osiris Ingestion Engine on {len(payloads)} repositories...")
    
    # This runs the multi-dimensional feature extraction, document scoring, and vectors
    ingest_results = ingest_batch(payloads, verbose=True)
    print_batch_summary(ingest_results)

    # ---------------------------------------------------------
    # PREPARE FOR LAYER 3
    # ---------------------------------------------------------
    # We must bridge the outputs of Ingestion into the storage schemas expected 
    # by Candidate Retrieval. In production, Postgres & Qdrant would serve as the bridge.
    
    records = []
    vectors = {}
    metadata_store = {}
    
    for payload, ingestion_res in zip(payloads, ingest_results):
        repo_id = ingestion_res.repo_id
        
        # Simulate a 384-dimensional vector (since Ingestion Engine computes this internally and sends directly to Qdrant)
        vectors[repo_id] = [0.1] * 384 
        metadata_store[repo_id] = {"description": "Ingested repo data"}

        # Map ingestion outputs to the Postgres schema expected by Retrieval
        records.append(RepositoryRecord(
            repo_id=repo_id,
            full_name=repo_id,
            description=payload.get("description"),
            topics=ingestion_res.tags,
            languages=[payload.get("primary_language")] if payload.get("primary_language") else [],
            stars=payload.get("star_count", 10),
            quality_score=ingestion_res.doc_quality,
            novelty_score=ingestion_res.novelty.final,
            activity_score=ingestion_res.activity_score,
            created_at=datetime.now(timezone.utc)
        ))
        
    print(f"\nBridged {len(records)} records from Ingestion into Retrieval format.")

    # ---------------------------------------------------------
    # LAYER 3: CANDIDATE RETRIEVAL
    # ---------------------------------------------------------
    print("\n=== [3] CANDIDATE RETRIEVAL LAYER ===")
    
    # 1. Mount data into InMemory Adapters (simulating Postgres, Qdrant, and Redis)
    metadata_repo = InMemoryMetadataRepository(records)
    vector_repo = InMemoryVectorRepository(vectors, metadata_store)
    cache = InMemoryAsyncCache()
    
    # 2. Setup a mock User Persona for testing recommendations
    user_id = "demo-user-123"
    persona = UserPersona(
        embedding=[0.1] * 384,  # Close to our mock vectors, meaning semantic search should match
        language_scores={"python": 1.0},
        interest_scores={"machine learning": 0.8},
        exploration_score=0.15
    )
    persona_store = InMemoryUserPersonaStore({user_id: persona})

    # 3. Initialize the Retriever Orchestrator
    retriever = CandidateRetriever(
        persona_store=persona_store,
        vector_repository=vector_repo,
        metadata_repository=metadata_repo,
        cache=cache
    )
    
    print(f"Running recommendation retrieval for {user_id}...")
    candidates = await retriever.retrieve(user_id, limit=5)
    
    print(f"\nSUCCESS! Retrieved {len(candidates)} recommended candidates:")
    for c in candidates:
        print(f" -> {c.repo_id:<30} | Score: {c.retrieval_score:.3f} | Sources: {c.retrieval_source}")
        
    print("\nEnd-to-End Pipeline test complete.")


if __name__ == "__main__":
    # Force some environment variables to prevent actual DB writes if Ingestion engine tries them
    # though ingestion_engine defaults to in-memory Qdrant if credentials are missing
    asyncio.run(run_end_to_end_test())

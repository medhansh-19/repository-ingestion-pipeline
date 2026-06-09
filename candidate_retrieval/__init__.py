"""Candidate Retrieval Layer for repository recommendations."""

from .adapters import (
    AsyncpgMetadataRepository,
    InMemoryMetadataRepository,
    InMemoryUserPersonaStore,
    InMemoryVectorRepository,
    QdrantVectorRepository,
)
from .cache import InMemoryAsyncCache, RedisAsyncCache
from .models import CandidateRepo, RepositoryRecord, UserPersona, VectorSearchHit
from .retriever import CandidateRetriever, merge_candidates

__all__ = [
    "AsyncpgMetadataRepository",
    "CandidateRepo",
    "CandidateRetriever",
    "InMemoryAsyncCache",
    "InMemoryMetadataRepository",
    "InMemoryUserPersonaStore",
    "InMemoryVectorRepository",
    "QdrantVectorRepository",
    "RedisAsyncCache",
    "RepositoryRecord",
    "UserPersona",
    "VectorSearchHit",
    "merge_candidates",
]

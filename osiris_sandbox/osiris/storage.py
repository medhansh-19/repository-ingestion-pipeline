"""
Layer 4: Candidate Retrieval, Multi-Dimensional Novelty, and Storage.
Handles vector database interactions, nearest neighbor candidate retrieval, 
and matrix thresholding logic.
"""

from __future__ import annotations
import uuid
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    _QDRANT_OK = True
except ImportError:
    _QDRANT_OK = False

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
COLLECTION_NAME = "osiris_research_corpus"
TOP_K_COMPARISONS = 5
NOVELTY_THRESHOLD = 0.35
VELOCITY_THRESHOLD = 0.40
DUPLICATE_SIMILARITY_THRESHOLD = 0.94
WRAPPER_SIMILARITY_THRESHOLD = 0.85

NOVELTY_WEIGHTS = {
    "semantic": 0.60,
    "tech_stack": 0.20,
    "category": 0.10,
    "activity": 0.10,
}

# Module-level state tracking
_qdrant_instance: Optional[QdrantClient] = None
_internal_corpus_history: List[dict] = []  # Fallback in-memory store
_growth_timeline_stream: List[dict] = []

@dataclass
class NoveltyResult:
    final: float
    semantic: float
    tech_stack: float
    category: float
    activity_dim: float
    top_k: List[dict]
    corpus_size: int
    anomaly_tag: str
    explanation: str

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def calculate_quadrant(novelty_score: float, velocity_score: float) -> Tuple[str, str]:
    """Maps a repository to a 2x2 matrix quadrant."""
    if novelty_score >= NOVELTY_THRESHOLD:
        if velocity_score >= VELOCITY_THRESHOLD:
            return "🔥 Viral Rockets", "APPROVED"
        else:
            return "💎 Hidden Gems", "APPROVED"
    else:
        if velocity_score >= VELOCITY_THRESHOLD:
            return "⚠️ Copycats / Clones", "REJECTED"
        else:
            return "💤 Dormant Ecosystem Nodes", "REJECTED"

# ═══════════════════════════════════════════════════════════════════
# CANDIDATE RETRIEVAL LAYER
# ═══════════════════════════════════════════════════════════════════
def retrieve_candidates(
    query_vector: np.ndarray,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
) -> Tuple[List[dict], int]:
    """
    Core Retrieval: Fetches Top 20 nearest neighbors from Qdrant.
    Fails over seamlessly to in-memory NumPy scanning if the DB is offline.
    """
    global _qdrant_instance
    qdrant_hits = []
    corpus_total = len(_internal_corpus_history)
    
    if _QDRANT_OK:
        try:
            if _qdrant_instance is None:
                _qdrant_instance = QdrantClient(url=qdrant_url, api_key=qdrant_api_key) if qdrant_url else QdrantClient(":memory:")
            
            # Verify collection exists
            collections = {c.name for c in _qdrant_instance.get_collections().collections}
            if COLLECTION_NAME not in collections:
                _qdrant_instance.create_collection(
                    COLLECTION_NAME,
                    vectors_config=VectorParams(size=query_vector.shape[0], distance=Distance.COSINE),
                )
            
            corpus_total = _qdrant_instance.count(COLLECTION_NAME).count
            
            if corpus_total > 0:
                search_result = _qdrant_instance.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector.tolist(),
                    limit=20,
                    with_payload=True
                ).points
                
                for hit in search_result:
                    qdrant_hits.append({
                        "repo_id": hit.payload.get("repo_id", ""),
                        "sim": hit.score,
                        "category": hit.payload.get("category", "General / Other"),
                        "activity": hit.payload.get("activity", 0.5),
                        "tags": hit.payload.get("tags", [])
                    })
        except Exception as e:
            print(f"\n[CRITICAL DB ERROR] Qdrant Search Failed: {e}")

    # Fallback Retrieval (Memory Scan)
    if not qdrant_hits and corpus_total > 0:
        for node in _internal_corpus_history:
            sim = _cosine(query_vector, node["vector"])
            qdrant_hits.append({
                "repo_id": node["repo_id"],
                "sim": sim,
                "category": node["category"],
                "activity": node["activity"],
                "tags": node["tags"]
            })
        qdrant_hits = sorted(qdrant_hits, key=lambda x: x["sim"], reverse=True)[:20]

    return qdrant_hits, max(corpus_total, len(_internal_corpus_history))

# ═══════════════════════════════════════════════════════════════════
# NOVELTY EVALUATION & STORAGE
# ═══════════════════════════════════════════════════════════════════
def compute_novelty(
    repo_id: str,
    target_vector: np.ndarray,
    target_cat: str,
    activity_val: float,
    target_tags: List[str]
) -> NoveltyResult:
    """
    Executes Candidate Retrieval and computes Multi-Dimensional Novelty against the slice.
    """
    candidates, corpus_volume = retrieve_candidates(target_vector)
    
    if corpus_volume == 0:
        return NoveltyResult(
            final=1.0, semantic=1.0, tech_stack=1.0, category=1.0, activity_dim=1.0,
            top_k=[], corpus_size=0, anomaly_tag="NONE",
            explanation="Initial baseline seed node — 100% processing system novelty clear."
        )

    # 1. Strict Self-Match Filtering on Top 20 Candidates
    filtered_neighbors = [hit for hit in candidates if hit["repo_id"] != repo_id]
    nearest_neighbors = filtered_neighbors[:TOP_K_COMPARISONS]

    if not nearest_neighbors:
        return NoveltyResult(
            final=1.0, semantic=1.0, tech_stack=1.0, category=1.0, activity_dim=1.0,
            top_k=[], corpus_size=corpus_volume, anomaly_tag="NONE",
            explanation="Retrieved Qdrant candidates matched self-instance mappings only."
        )

    # 2. Semantic Novelty
    peak_similarity = nearest_neighbors[0]["sim"]
    mean_neighbor_similarity = sum(node["sim"] for node in nearest_neighbors) / len(nearest_neighbors)
    semantic_novelty = round(1.0 - mean_neighbor_similarity, 4)

    # 3. Tech-Stack Overlap (Jaccard)
    neighbor_tags = set(nearest_neighbors[0]["tags"])
    active_tags = set(target_tags)
    if active_tags or neighbor_tags:
        shared = active_tags.intersection(neighbor_tags)
        total = active_tags.union(neighbor_tags)
        tech_stack_novelty = round(1.0 - (len(shared) / len(total)), 4)
    else:
        tech_stack_novelty = 0.50

    # 4. Category Saturation
    matching_cat_count = sum(1 for node in filtered_neighbors if node.get("category") == target_cat)
    slice_size = len(filtered_neighbors) or 1
    category_novelty = round(1.0 - min((matching_cat_count / slice_size) * 1.5, 1.0), 4)

    # 5. Relative Activity Deviation
    active_energies = [node.get("activity", 0.5) for node in filtered_neighbors]
    mean_energy = sum(active_energies) / len(active_energies) if active_energies else 0.5
    activity_novelty = round(min(activity_val / max(mean_energy, 0.01), 1.0), 4)

    # Matrix Fusion
    final_score = round(
        NOVELTY_WEIGHTS["semantic"] * semantic_novelty +
        NOVELTY_WEIGHTS["tech_stack"] * tech_stack_novelty +
        NOVELTY_WEIGHTS["category"] * category_novelty +
        NOVELTY_WEIGHTS["activity"] * activity_novelty,
        4
    )

    # Anomaly Detection
    anomaly = "NONE"
    if peak_similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
        anomaly = "POTENTIAL_FORK_OR_DIRECT_CLONE"
    elif peak_similarity >= WRAPPER_SIMILARITY_THRESHOLD and tech_stack_novelty < 0.20:
        anomaly = "POTENTIAL_SHALLOW_API_WRAPPER"

    # Explanation Mapping
    explanation_logs = ["Ecosystem Structural Similarity Trace (Top Candidates):"]
    for position, match in enumerate(nearest_neighbors[:3], 1):
        explanation_logs.append(f"  [{position}] {match['repo_id'].split('/')[-1]:<24} Cosine Sim: {match['sim']:.4f} | Taxonomy: {match['category']}")
        
    explanation_logs += [
        "",
        "Calculated Dimensional Matrix Sub-Novelty Indexes:",
        f"  >> Semantic Vector Distance     : {semantic_novelty:.4f}  [Weight: 60%]",
        f"  >> Tech-Stack Jaccard Delta     : {tech_stack_novelty:.4f}  [Weight: 20%]",
        f"  >> Domain Cluster Saturation    : {category_novelty:.4f}  [Weight: 10%]",
        f"  >> Activity Energy Ratio        : {activity_novelty:.4f}  [Weight: 10%]",
        f"  ───────────────────────────────────────────────────────────────────",
        f"  [*] Consolidated Novelty Engine Index Output ===> {final_score:.4f}"
    ]

    return NoveltyResult(
        final=final_score, semantic=semantic_novelty, tech_stack=tech_stack_novelty,
        category=category_novelty, activity_dim=activity_novelty,
        top_k=nearest_neighbors, corpus_size=corpus_volume, anomaly_tag=anomaly,
        explanation="\n".join(explanation_logs)
    )

def store_approved_candidate(repo_id: str, vector: np.ndarray, metadata: dict):
    """Upserts an approved candidate into Qdrant and fallback memory."""
    global _qdrant_instance
    _internal_corpus_history.append({
        "repo_id": repo_id,
        "vector": vector,
        "category": metadata.get("category"),
        "activity": metadata.get("activity"),
        "tags": metadata.get("tags")
    })
    
    _growth_timeline_stream.append({
        "growth_index": len(_internal_corpus_history),
        "repo_id": repo_id,
        "novelty_index_point": metadata.get("novelty_score", 0.0)
    })
    
    if _QDRANT_OK and _qdrant_instance:
        try:
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_id))
            _qdrant_instance.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(id=point_uuid, vector=vector.tolist(), payload={"repo_id": repo_id, **metadata})
                ]
            )
        except Exception:
            pass
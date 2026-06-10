
"""
Osiris — Advanced Repository Ingestion & Novelty Evaluation Engine (v2.0)
===========================================================================
A research-grade pipeline designed to analyze incoming software repositories,
extract multi-dimensional features, evaluate ecosystem novelty, map cluster
topologies, and execute analytical ingestion decisions.

Pipeline Architecture:
    staged_repo (JSON)
         │
         ├─► [1] Tag Extraction & Structural Token Normalization
         ├─► [2] Documentation Quality Scoring (Multi-Section Validation)
         ├─► [3] Activity Index & Exponential Decay Normalization
         ├─► [4] Trend Velocity Scoring (3d / 7d / 30d Blended Flux)
         ├─► [5] Standardized Structured Summary Generation (Zero-LLM)
         ├─► [6] Taxonomic Rule-Based Category Classification 
         ├─► [7] Hybrid Semantic Representation Vector Fusion
         │         Fuses aligned structural sub-vectors across 5 clean signals:
         │         0.40 * Readme + 0.25 * Summary + 0.20 * Topics + 0.10 * Cat + 0.05 * Lang
         ├─► [8] Dynamic Vector Centroid Cluster & Space Discovery
         ├─► [9] Multi-Dimensional Novelty Matrix (Strict Self-Match Excluded)
         │         0.60 * Semantic + 0.20 * Tech-Stack + 0.10 * Category + 0.10 * Activity
         ├─► [10] Ecosystem Anomaly, Clone, and Wrapper Template Detection Flags
         ├─► [11] Structural Explainability Logic & Historical Timeline Decay Tracker
         └─► [12] Automated Ingestion Gateway Decision & Full Corpus Summary Visualizer

Requirements:
    pip install qdrant-client sentence-transformers numpy
"""

from __future__ import annotations
import os
import sys
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple

# Core Numeric Vector Stack Validation
try:
    import numpy as np
except ImportError:
    print("Execution Error: 'numpy' library is required. Install via: pip install numpy")
    sys.exit(1)

# Vector Database Engine Verification
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    try:
        from qdrant_client.models import NamedVector
    except ImportError:
        NamedVector = None
    _QDRANT_OK = True
except ImportError:
    _QDRANT_OK = False

# Local Deep Learning Inference Stack Verification
try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except ImportError:
    _ST_OK = False


# ═══════════════════════════════════════════════════════════════════
# GLOBAL SYSTEM DESIGN CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
NOVELTY_THRESHOLD               = 0.35
TOP_K_COMPARISONS               = 5
EMBEDDING_MODEL                 = "all-MiniLM-L6-v2"
EMBEDDING_DIM                   = 384
COLLECTION_NAME                 = "osiris_research_corpus"
QDRANT_VECTOR_NAME              = "repo_embedding"
MAX_DOC_SCORE                   = 100

# Anomaly & Integrity Detection Thresholds
DUPLICATE_SIMILARITY_THRESHOLD  = 0.94
WRAPPER_SIMILARITY_THRESHOLD    = 0.85

# Hybrid semantic weight configurations (Vector coefficients must sum to 1.0)
HYBRID_WEIGHTS = {
    "readme":       0.40,
    "description":  0.25,
    "topics":       0.20,
    "category":     0.10,
    "language":     0.05,
}

# Multi-dimensional evaluation weights (Matrix elements must sum to 1.0)
NOVELTY_WEIGHTS = {
    "semantic":   0.60,
    "tech_stack": 0.20,
    "category":   0.10,
    "activity":   0.10,
}


# ═══════════════════════════════════════════════════════════════════
# 1. TAG EXTRACTION & METADATA CLEANING
# ═══════════════════════════════════════════════════════════════════
def extract_tags(repo_id: str, paragraphs: list[str]) -> list[str]:
    """Extracts structural tokens from the title and semantic phrases from docs."""
    stop_words = {
        'app', 'api', 'repo', 'project', 'demo', 'test', 'tool', 'my', 'the',
        'and', 'for', 'with', 'io', 'github', 'boilerplate', 'template', 'version'
    }
    
    # 1. Extract from Title Name
    t = repo_id.split("/")[-1]
    for pat, rep in {
        r'REST(ful)?': lambda m: 'Rest' + (m.group(1) or ''),
        r'JWT': 'Jwt', r'GraphQL': 'Graphql', r'MySQL': 'Mysql',
        r'PostgreSQL': 'Postgresql', r'NoSQL': 'Nosql',
    }.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        
    t = re.sub(r'(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])', 'cpp', t, flags=re.IGNORECASE)
    t = re.sub(r'(?<![a-zA-Z0-9])c#(?![a-zA-Z0-9])',    'csharp', t, flags=re.IGNORECASE)
    t = re.sub(r'\.js\b', ' js', t, flags=re.IGNORECASE)
    t = re.sub(r'\.ts\b', ' ts', t, flags=re.IGNORECASE)
    t = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', t)
    t = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', t)
    t = re.sub(r'([a-z0-9])([A-Z])',  r'\1 \2', t)
    t = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', t)
    t = re.sub(r'[-_./\\:|+=#@^&*~`?<>!]', ' ', t)
    
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', t)
    seen = set()
    tags = [
        tok.lower() for tok in tokens
        if tok.lower() not in stop_words and not (seen.add(tok.lower()) or tok.lower() in seen)
    ]
    
    # 2. Extract compound semantic concepts from descriptions
    text_corpus = " ".join(paragraphs).lower()
    high_value_phrases = [
        "ai assistant", "tool calling", "local ai", "voice interaction", "voice",
        "autonomous", "agent", "llm", "rag", "vector database", "machine learning",
        "inference", "multi-agent", "frontend", "backend"
    ]
    
    for phrase in high_value_phrases:
        if phrase in text_corpus and phrase not in tags:
            tags.append(phrase.title())
            
    return tags


# ═══════════════════════════════════════════════════════════════════
# 2. DOCUMENTATION QUALITY SCORING
# ═══════════════════════════════════════════════════════════════════
_SECTION_SCORES = {
    "installation": 15, "usage": 15, "api": 10,
    "contributing": 10, "license": 10, "faq": 5,
}
_SECTION_RE = {
    "installation": re.compile(r"\b(installation|install|setup|getting[\s_]started|quick[\s_]start)\b", re.I),
    "usage":        re.compile(r"\b(usage|examples?|how[\s_]to[\s_]use|quickstart)\b", re.I),
    "api":          re.compile(r"\b(api[\s_]?reference|api[\s_]?docs?|endpoints?)\b", re.I),
    "contributing": re.compile(r"\b(contributing|contribution[s]?)\b", re.I),
    "license":      re.compile(r"\b(licen[sc]e|mit|apache|gpl)\b", re.I),
    "faq":          re.compile(r"\b(faq|frequently[\s_]asked|troubleshoot)\b", re.I),
}

@dataclass
class DocResult:
    score: float          
    raw: int
    found: list[str]
    missing: list[str]
    breakdown: dict

def score_documentation(repo: dict) -> DocResult:
    """Evaluates the completeness and distribution structural depth of the README text."""
    readme_len = repo.get("readme_length", 0)
    ratio      = repo.get("readme_to_codebase_ratio", 0.0)
    text       = " ".join(repo.get("extracted_paragraphs", []))
    
    pts = 0
    bd  = {}
    if readme_len > 0:
        pts += 10; bd["readme_exists"] = 10
    if readme_len > 500:  
        pts += 10; bd["length_500"]    = 10
    if readme_len > 2000: 
        pts +=  5; bd["length_2000"]   = 5
    if ratio > 0.001:     
        pts +=  5; bd["ratio_bonus"]   = 5
        
    found, missing = [], []
    for sec, rx in _SECTION_RE.items():
        if rx.search(text):
            pts += _SECTION_SCORES[sec]
            bd[sec] = _SECTION_SCORES[sec]
            found.append(sec)
        else:
            missing.append(sec)
            
    raw = min(pts, MAX_DOC_SCORE)
    return DocResult(score=round(raw / MAX_DOC_SCORE, 4), raw=raw, found=found, missing=missing, breakdown=bd)


# ═══════════════════════════════════════════════════════════════════
# 3. OPEN-SOURCE ECOSYSTEM ACTIVITY INDEX
# ═══════════════════════════════════════════════════════════════════
def activity_score(repo: dict) -> float:
    """Calculates active code maintainer energy incorporating real-time exponential half-life decay."""
    pushed_days = repo.get("pushed_days_ago", 999)
    recency  = math.exp(-pushed_days * math.log(2) / 30.0) # 30-day half life scale
    stars    = min(math.log1p(repo.get("star_count", 0)) / math.log1p(200_000), 1.0)
    contrib  = min(repo.get("mentionable_users_count", 0) / 10.0, 1.0)
    return round(0.50 * recency + 0.35 * stars + 0.15 * contrib, 4)


# ═══════════════════════════════════════════════════════════════════
# 4. STAR FLUX TREND VELOCITY
# ═══════════════════════════════════════════════════════════════════
def trend_velocity(repo: dict) -> float:
    """Quantifies current social popularity traction across multi-stage window intervals."""
    r3  = repo.get("delta_3d",  0) / 3.0
    r7  = repo.get("delta_7d",  0) / 7.0
    r30 = repo.get("delta_30d", 0) / 30.0
    
    blend = 0.50 * r3 + 0.30 * r7 + 0.20 * r30
    vel   = min(math.log1p(blend) / math.log1p(500), 1.0)
    
    # Acceleration boost flag for sudden exponential breakouts
    if r30 > 0 and r3 > r30:
        vel = min(vel + 0.15 * min((r3 - r30) / max(r30, 1), 1.0), 1.0)
    return round(vel, 4)


# ═══════════════════════════════════════════════════════════════════
# 5. STRUCTURED REGEX-BASED SUMMARY GENERATOR
# ═══════════════════════════════════════════════════════════════════
def build_structured_summary(repo: dict, tags: list[str], category: str) -> str:
    """
    Standardizes variable-length documentation into a structured presentation blueprint.
    Ensures that evaluating systems maintain feature-weight normalization parity.
    """
    name = repo.get("id", "unknown/repo").split("/")[-1]
    lang = repo.get("primary_language", "Unknown") or "Unknown"
    stars = repo.get("star_count", 0)
    paras = repo.get("extracted_paragraphs", [])
    
    # Process text layout abstracts
    clean_paras = [re.sub(r"<[^>]+>", " ", p).strip() for p in paras]
    clean_paras = [re.sub(r"\s+", " ", p) for p in clean_paras if len(p) > 30]
    
    abstract = max(clean_paras, key=len) if clean_paras else "(No core architectural documentation synopsis available)"
    if len(abstract) > 300:
        abstract = abstract[:300].rsplit(" ", 1)[0] + "..."

    lines = [
        f"Repository: {name}",
        f"Category: {category}",
        f"Primary Language: {lang}",
        f"Stars: {stars:,}",
        "",
        "Core Purpose:",
        f"  {abstract}",
        "",
        "Technology Stack / Extracted Tags:",
        f"  {', '.join(tags[:12]) if tags else 'No semantic keywords flagged'}"
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 6. DOMAIN EXPERT TAXONOMY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
_TAXONOMY_RULES: list[tuple[str, list[str]]] = [
    ("AI Agent",            ["agent", "autonomous", "multi-agent", "agentic", "autogpt", "crewai", "opendevin", "babyagi", "react", "openclaw", "hermes", "opencode", "assistant", "tool calling", "voice interaction"]),    ("LLM / Foundation",    ["llm", "gpt", "claude", "gemini", "llama", "mistral", "transformer", "bert", "embeddings", "rag", "vector"]),
    ("ML Library",          ["pytorch", "tensorflow", "jax", "keras", "sklearn", "scikit", "neural", "deep learning", "training", "inference"]),
    ("Developer Tool",      ["cli", "vscode", "ide", "debugger", "linter", "formatter", "profiler", "codegen", "homebrew", "git", "compiler"]),
    ("Web Framework",       ["flask", "django", "fastapi", "express", "rails", "laravel", "spring", "http", "middleware", "graphql"]),
    ("Frontend",            ["react", "vue", "svelte", "angular", "nextjs", "tailwind", "css", "ui", "component", "design system", "chart.js"]),
    ("Infrastructure",      ["kubernetes", "docker", "terraform", "ansible", "helm", "devops", "cicd", "cloud", "aws", "serverless", "nginx"]),
    ("Data Engineering",    ["spark", "kafka", "airflow", "dbt", "pipeline", "etl", "warehouse", "lakehouse", "streaming", "hadoop", "flink"]),
    ("Security",            ["auth", "jwt", "oauth", "cryptography", "encryption", "vulnerability", "pentest", "firewall", "biometric", "keystroke"]),
    ("Blockchain",          ["solidity", "ethereum", "bitcoin", "defi", "nft", "web3", "smart contract", "crypto", "ledger", "solana"]),
    ("Automation",          ["automation", "scraping", "workflow", "n8n", "zapier", "rpa", "bot", "scheduler", "selenium", "playwright"]),
    ("Systems / Low-level", ["rust", "zig", "assembly", "kernel", "llvm", "memory", "embedded", "firmware", "c++", "clang", "operating system"])
]

def classify_category(repo: dict, tags: list[str]) -> str:
    """Dispatches precise software classification tags matching structural keywords."""
    text_corpus = " ".join([
        repo.get("id", ""),
        repo.get("primary_language", ""),
        " ".join(repo.get("extracted_paragraphs", [])),
        " ".join(tags)
    ]).lower()
    
    selected_category, highest_hits = "General / Other", 0
    for category, keywords in _TAXONOMY_RULES:
        hits = sum(1 for kw in keywords if kw in text_corpus)
        if hits > highest_hits:
            highest_hits = hits
            selected_category = category
            
    return selected_category


# ═══════════════════════════════════════════════════════════════════
# 7. HIGH-FIDELITY HYBRID VECTOR SPACE EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════
_fallback_vocab: dict[str, int] = {}
_FALLBACK_DIM = 384  # Aligned to matches Sentence Transformer outputs perfectly

def _generate_fallback_vector(text: str) -> np.ndarray:
    """Mathematical fallback string hash distribution mapping if SentenceTransformer fails."""
    words = re.findall(r"[a-z]{2,}", text.lower())
    vector = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    for word in words:
        if word not in _fallback_vocab:
            if len(_fallback_vocab) < _FALLBACK_DIM:
                _fallback_vocab[word] = len(_fallback_vocab)
            else:
                continue
        vector[_fallback_vocab[word]] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector

_loaded_transformer = None
def _generate_embedding(text: str) -> np.ndarray:
    """Generates normalized vector matrices using active neural pipeline topologies."""
    global _loaded_transformer
    if _ST_OK:
        try:
            if _loaded_transformer is None:
                _loaded_transformer = SentenceTransformer(EMBEDDING_MODEL)
            dense_vector = _loaded_transformer.encode(text, normalize_embeddings=True)
            return dense_vector.astype(np.float32)
        except Exception:
            pass
    return _generate_fallback_vector(text)

def generate_hybrid_embedding(repo: dict, tags: list[str], category: str, summary: str) -> np.ndarray:
    """
    Executes a Multi-Field Component Vector Fusion Model.
    Each feature field is embedded within its own localized latent context window 
    before linear combination via static projection coefficients to maintain vector space properties.
    """
    paragraphs = repo.get("extracted_paragraphs", [])
    raw_readme = " ".join(re.sub(r"<[^>]+>", " ", p) for p in paragraphs)[:1500]
    primary_lang = repo.get("primary_language", "unknown")
    repo_name = repo.get("id", "unknown/repo").split("/")[-1]

    # Distinct separate functional spaces
    fields = {
        "readme":      raw_readme,
        "description": summary,
        "topics":      " ".join(tags) + f" {repo_name}",
        "category":    category,
        "language":    primary_lang
    }

    # Generate isolated sub-vectors
    sub_vectors = {k: _generate_embedding(v if v.strip() else "missing") for k, v in fields.items()}
    
    # Weighted integration framework execution (Maintains geometric alignment bounds)
    fused_matrix = np.zeros(_FALLBACK_DIM, dtype=np.float32)
    for key, coefficient in HYBRID_WEIGHTS.items():
        fused_matrix += coefficient * sub_vectors[key]
        
    final_norm = np.linalg.norm(fused_matrix)
    return (fused_matrix / final_norm if final_norm > 0 else fused_matrix).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 8. QDRANT NATIVE STORAGE INTERFACE SERVICE
# ═══════════════════════════════════════════════════════════════════
_qdrant_instance: Optional[object] = None

def _resolve_qdrant_client(url=None, api_key=None):
    global _qdrant_instance
    if _qdrant_instance is None and _QDRANT_OK:
        url = url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        if url:
            _qdrant_instance = QdrantClient(url=url, api_key=api_key)
        else:
            _qdrant_instance = QdrantClient(":memory:")
    return _qdrant_instance

def _verify_qdrant_collection(client, target_dim: int):
    try:
        collections = {c.name for c in client.get_collections().collections}
        if COLLECTION_NAME not in collections:
            client.create_collection(
                COLLECTION_NAME,
                vectors_config={QDRANT_VECTOR_NAME: VectorParams(size=target_dim, distance=Distance.COSINE)},
            )
            print(f"[QDRANT] Created collection={COLLECTION_NAME} vector_name={QDRANT_VECTOR_NAME} vector_dim={target_dim} distance=COSINE")
            return

        diagnostics = _qdrant_collection_diagnostics(client)
        configured_dim = diagnostics.get("vector_size")
        if configured_dim and configured_dim != target_dim:
            print(
                f"[QDRANT WARNING] Collection vector dimension mismatch: "
                f"collection_dim={configured_dim} query_dim={target_dim}. "
                f"Neighbor retrieval may return zero results until the collection is rebuilt."
            )
    except Exception as e:
        print(f"[QDRANT WARNING] Collection verification failed: {e}")

def _index_qdrant_point(client, repo_id: str, vector: np.ndarray, metadata: dict):
    try:
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, repo_id))
        vector_payload = _qdrant_point_vector(client, vector)
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(id=point_uuid, vector=vector_payload, payload={"repo_id": repo_id, **metadata})
            ],
            wait=True,
        )
        collection_size = _qdrant_count_points(client)
        print(f"[QDRANT] Inserted point_id={point_uuid} repo_id={repo_id} collection_size={collection_size}")
    except Exception as e:
        print(f"\n[CRITICAL DB ERROR] Qdrant Insert Failed for {repo_id}: {e}")

def _qdrant_collection_diagnostics(client) -> dict:
    """Returns best-effort collection details across qdrant-client versions."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        vector_size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        vector_names = []
        if isinstance(vectors, dict):
            vector_names = list(vectors.keys())
            selected = vectors.get(QDRANT_VECTOR_NAME) or next(iter(vectors.values()), None)
            vector_size = getattr(selected, "size", None)
            distance = getattr(selected, "distance", None)
        return {
            "vector_size": vector_size,
            "vector_names": vector_names,
            "vector_mode": "named" if vector_names else "unnamed",
            "distance": str(distance) if distance is not None else None,
            "points_count": getattr(info, "points_count", None),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
        }
    except Exception as e:
        return {"error": str(e)}

def _qdrant_count_points(client) -> int:
    try:
        return int(client.count(COLLECTION_NAME, exact=True).count)
    except TypeError:
        return int(client.count(COLLECTION_NAME).count)
    except Exception:
        return 0

def _qdrant_uses_named_vectors(client) -> bool:
    diagnostics = _qdrant_collection_diagnostics(client)
    return diagnostics.get("vector_mode") == "named"

def _qdrant_point_vector(client, vector: np.ndarray):
    if _qdrant_uses_named_vectors(client):
        return {QDRANT_VECTOR_NAME: vector.tolist()}
    return vector.tolist()

def _qdrant_query_vector(client, vector: np.ndarray):
    if _qdrant_uses_named_vectors(client):
        if NamedVector is not None:
            return NamedVector(name=QDRANT_VECTOR_NAME, vector=vector.tolist())
        return {"name": QDRANT_VECTOR_NAME, "vector": vector.tolist()}
    return vector.tolist()

def _query_qdrant_neighbors(client, query_vector: np.ndarray, *, limit: int = 20) -> list[dict]:
    """Queries Qdrant with compatibility fallbacks and payload diagnostics."""
    hits = []
    try:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=_qdrant_query_vector(client, query_vector),
            limit=limit,
            with_payload=True,
        )
        points = getattr(search_result, "points", search_result)
    except Exception as query_error:
        print(f"[QDRANT WARNING] query_points failed, trying legacy search API: {query_error}")
        try:
            points = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=_qdrant_query_vector(client, query_vector),
                limit=limit,
                with_payload=True,
            )
        except Exception as search_error:
            print(f"[QDRANT WARNING] Legacy search API also failed: {search_error}")
            return []

    for hit in points or []:
        payload = getattr(hit, "payload", None) or {}
        hits.append({
            "repo_id": payload.get("repo_id", ""),
            "sim": float(getattr(hit, "score", 0.0)),
            "category": payload.get("category", "General / Other"),
            "activity": payload.get("activity", 0.5),
            "tags": payload.get("tags", []),
        })
    return hits

def _query_internal_neighbors(query_vector: np.ndarray, *, limit: int = 20) -> list[dict]:
    """Fallback ANN scan over in-process corpus history."""
    internal_hits = []
    for node in _internal_corpus_history:
        internal_hits.append({
            "repo_id": node["repo_id"],
            "sim": _cosine(query_vector, node["vector"]),
            "category": node.get("category", "General / Other"),
            "activity": node.get("activity", 0.5),
            "tags": node.get("tags", []),
        })
    internal_hits.sort(key=lambda hit: hit["sim"], reverse=True)
    return internal_hits[:limit]


# ═══════════════════════════════════════════════════════════════════
# 9. RESEARCH FEATURE MODULES (Centroid Discovery & Analytics Tracker)
# ═══════════════════════════════════════════════════════════════════
_internal_corpus_history: list[dict] = []
_growth_timeline_stream: list[dict] = []

def dynamic_cluster_discovery(corpus: list[dict]) -> dict:
    """
    Executes real-time mathematical centroid tracking across incoming vector configurations.
    Measures dimensional saturation density distributions and topological boundary distances.
    """
    if len(corpus) < 4:
        return {"current_cluster": "System Bootstrapping Node", "nearest_cluster": "None", "distance": 0.0}
        
    from collections import Counter
    categories = [node["category"] for node in corpus]
    distribution = Counter(categories)
    
    top_cluster = distribution.most_common(1)[0][0]
    alt_cluster = distribution.most_common(2)[1][0] if len(distribution) > 1 else "External Ecosystem Hub"
    
    # Calculate dimensional centroid arrays
    vectors_top = [node["vector"] for node in corpus if node["category"] == top_cluster]
    vectors_alt = [node["vector"] for node in corpus if node["category"] == alt_cluster] if len(distribution) > 1 else vectors_top
    
    centroid_top = np.mean(vectors_top, axis=0)
    centroid_alt = np.mean(vectors_alt, axis=0)
    
    norm_t = np.linalg.norm(centroid_top)
    norm_a = np.linalg.norm(centroid_alt)
    
    cosine_distance = 1.0 - float(np.dot(centroid_top, centroid_alt) / (norm_t * norm_a)) if norm_t and norm_a else 1.0
    return {
        "current_cluster": top_cluster,
        "nearest_cluster": alt_cluster,
        "distance": round(cosine_distance, 4)
    }


# ═══════════════════════════════════════════════════════════════════
# 10. MULTI-DIMENSIONAL NOVELTY ENGINE WITH SELF-MATCH EXCLUSION
# ═══════════════════════════════════════════════════════════════════
@dataclass
class NoveltyResult:
    final:            float
    semantic:         float
    tech_stack:       float
    category:         float
    activity_dim:     float
    top_k:            list[dict]
    corpus_size:      int
    anomaly_tag:      str
    explanation:      str

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def compute_multi_dimensional_novelty(
    repo_id:        str,
    target_cat:     str,
    activity_val:   float,
    target_tags:    list[str],
    qdrant_hits:    list[dict],    # ◄ Now receives Top 20 ANN hits from Qdrant
    corpus_volume:  int            # ◄ Total corpus size
) -> NoveltyResult:
    """
    Evaluates multi-stage ecological matrix distance metrics strictly against 
    the Top 20 ANN neighbors retrieved from Qdrant.
    """
    if corpus_volume == 0:
        return NoveltyResult(
            final=1.0, semantic=1.0, tech_stack=1.0, category=1.0, activity_dim=1.0,
            top_k=[], corpus_size=0, anomaly_tag="NONE",
            explanation="Initial baseline seed node — 100% processing system novelty clear."
        )

    # ── [1] STRICT SELF-MATCH FILTERING ON TOP 20 HITS ──
    filtered_neighbors = []
    for hit in qdrant_hits:
        if hit["repo_id"] == repo_id:
            continue
        filtered_neighbors.append(hit)

    nearest_neighbors = filtered_neighbors[:TOP_K_COMPARISONS]

    if not nearest_neighbors:
        return NoveltyResult(
            final=1.0, semantic=1.0, tech_stack=1.0, category=1.0, activity_dim=1.0,
            top_k=[], corpus_size=corpus_volume, anomaly_tag="NONE",
            explanation="Retrieved Qdrant neighbors matched self-instance mappings only."
        )

    # 1. Semantic Novelty Core Generation (Using Qdrant's Cosine Scores)
    peak_similarity = nearest_neighbors[0]["sim"]
    mean_neighbor_similarity = sum(node["sim"] for node in nearest_neighbors) / len(nearest_neighbors)
    semantic_novelty_index = round(1.0 - mean_neighbor_similarity, 4)

    # 2. Structural Tech-Stack Overlap (Jaccard Distance)
    neighbor_tags = set(nearest_neighbors[0]["tags"])
    active_tags   = set(target_tags)
    if active_tags or neighbor_tags:
        shared_features = active_tags.intersection(neighbor_tags)
        total_features  = active_tags.union(neighbor_tags)
        tech_stack_novelty_index = round(1.0 - (len(shared_features) / len(total_features)), 4)
    else:
        tech_stack_novelty_index = 0.50

    # 3. Category Saturation Vectoring (Localized to Top 20 slice)
    matching_category_count = sum(1 for node in filtered_neighbors if node.get("category") == target_cat)
    slice_size = len(filtered_neighbors) or 1
    saturation_ratio = matching_category_count / slice_size
    category_novelty_index = round(1.0 - min(saturation_ratio * 1.5, 1.0), 4)

    # 4. Relative Activity Deviation Metric (Against Top 20 slice)
    active_corpus_energies = [node.get("activity", 0.5) for node in filtered_neighbors]
    mean_ecosystem_energy = sum(active_corpus_energies) / len(active_corpus_energies) if active_corpus_energies else 0.5
    activity_novelty_index = round(min(activity_val / max(mean_ecosystem_energy, 0.01), 1.0), 4)

    # ── MULTI-DIMENSIONAL WEIGHTED MATRIX FUSION ──
    aggregated_novelty_index = round(
        NOVELTY_WEIGHTS["semantic"]   * semantic_novelty_index +
        NOVELTY_WEIGHTS["tech_stack"] * tech_stack_novelty_index +
        NOVELTY_WEIGHTS["category"]   * category_novelty_index +
        NOVELTY_WEIGHTS["activity"]   * activity_novelty_index,
        4
    )

    # ── [TIER 3] DUPLICATE, CLONE, AND WRAPPER DETECTOR ──
    detected_anomaly_signature = "NONE"
    if peak_similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
        detected_anomaly_signature = "POTENTIAL_FORK_OR_DIRECT_CLONE"
    elif peak_similarity >= WRAPPER_SIMILARITY_THRESHOLD and tech_stack_novelty_index < 0.20:
        detected_anomaly_signature = "POTENTIAL_SHALLOW_API_WRAPPER"

    # Compile Structured Explanation Logs
    explanation_logs = [
        "Ecosystem Structural Similarity Trace (Top Neighbors via Qdrant ANN):"
    ]
    for position, match in enumerate(nearest_neighbors[:3], 1):
        explanation_logs.append(
            f"  [{position}] {match['repo_id'].split('/')[-1]:<24} Cosine Similarity: {match['sim']:.4f} | Taxonomy: {match['category']}"
        )
        
    diagnostic_traits = []
    primary_conflict_match = nearest_neighbors[0]
    if primary_conflict_match["category"] == target_cat:
        diagnostic_traits.append(f"Identical technology domain space overlap ({target_cat})")
        
    intersecting_tokens = list(active_tags.intersection(set(primary_conflict_match["tags"])))
    if intersecting_tokens:
        diagnostic_traits.append(f"Shared architectural mechanisms: [{', '.join(intersecting_tokens[:3])}]")
    if primary_conflict_match["sim"] > 0.82:
        diagnostic_traits.append("Highly convergent implementation architecture structures detected")

    explanation_logs += [
        "",
        "Architectural Explainability Diagnostics:",
        f"  • {'; '.join(diagnostic_traits) if diagnostic_traits else 'No dangerous component pattern convergences flagged.'}",
        "",
        "Calculated Dimensional Matrix Sub-Novelty Indexes:",
        f"  >> Semantic Vector Distance     : {semantic_novelty_index:.4f}  [Weight: 60%]",
        f"  >> Tech-Stack Jaccard Delta     : {tech_stack_novelty_index:.4f}  [Weight: 20%]",
        f"  >> Domain Cluster Saturation    : {category_novelty_index:.4f}  [Weight: 10%]",
        f"  >> Activity Energy Ratio        : {activity_novelty_index:.4f}  [Weight: 10%]",
        f"  ───────────────────────────────────────────────────────────────────",
        f"  [*] Consolidated Novelty Engine Index Output ===> {aggregated_novelty_index:.4f}"
    ]

    return NoveltyResult(
        final=aggregated_novelty_index, semantic=semantic_novelty_index, tech_stack=tech_stack_novelty_index,
        category=category_novelty_index, activity_dim=activity_novelty_index,
        top_k=nearest_neighbors, corpus_size=corpus_volume, anomaly_tag=detected_anomaly_signature,
        explanation="\n".join(explanation_logs)
    )
    # ═══════════════════════════════════════════════════════════════════
# QUADRANT MATRIX CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
VELOCITY_THRESHOLD = 0.15  # Midpoint boundary for Trend Velocity flux

def calculate_quadrant(novelty_score: float, velocity_score: float) -> tuple[str, str]:
    """
    Maps a repository to a 2x2 matrix quadrant based on its Novelty and Velocity profiles.
    Returns a tuple of (Quadrant_Name, Operational_Action).
    """
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
# 11. INGESTION GATEWAY INTERFACE Blueprints
# ═══════════════════════════════════════════════════════════════════
@dataclass
class IngestionResult:
    repo_id:               str
    decision:              str       
    rejection_reason:      str    
    doc_quality:           float    
    activity_score:        float    
    trend_velocity:        float    
    novelty:               NoveltyResult
    tags:                  list[str]
    category:              str
    structured_summary:    str
    doc_found:             list[str]
    doc_missing:           list[str]
    topological_metrics:   dict = field(default_factory=dict)
    quadrant:              str  = "Unknown"

    def __str__(self) -> str:
        width = 76
        thin_divider = "─" * width
        thick_divider = "═" * width
        status_banner = "✅ REGISTRATION APPROVED" if self.decision == "APPROVED" else "❌ ECOSYSTEM DROP REJECTED"
        
        anomaly_warning = ""
        if self.novelty.anomaly_tag != "NONE":
            anomaly_warning = f"\n  ⚠️  SECURITY/INTEGRITY ANOMALY DETECTED: [{self.novelty.anomaly_tag}]"

        output_blocks = [
            thick_divider,
            f"  {status_banner}   ::   {self.repo_id}",
            f"  Assigned Taxonomy Core Module Node : {self.category}{anomaly_warning}",
            f"  Ecosystem Matrix Placement         : {self.quadrant}",
            thin_divider,
            f"  Documentation Quality : {self.doc_quality:.4f}  (Found Components: {', '.join(self.doc_found) or 'None'})",
            f"  Activity Density Index: {self.activity_score:.4f}",
            f"  Trend Flux Velocity   : {self.trend_velocity:.4f}",
            thin_divider,
            f"  Standardized Structural Presentation Abstract:",
            "\n".join(f"    {line}" for line in self.structured_summary.split("\n")),
            thin_divider,
            f"  Aggregate System Novelty Target Index: {self.novelty.final:.4f}  (Gateway Minimum Constraint Floor: ≥ {NOVELTY_THRESHOLD})",
            thin_divider,
            self.novelty.explanation
        ]
        
        if self.topological_metrics and self.topological_metrics.get("nearest_cluster") != "None":
            output_blocks += [
                thin_divider,
                f"  Corpus Cluster Topology Structural Space Vector:",
                f"    Core Target Centroid Node : {self.topological_metrics.get('current_cluster')}",
                f"    Adjacent Cluster Center   : {self.topological_metrics.get('nearest_cluster')}",
                f"    Spatial Centroid Distance : {self.topological_metrics.get('distance'):.4f}"
            ]
            
        if self.rejection_reason:
            output_blocks += ["", f"  Rejection Gate Analysis Logic: {self.rejection_reason}"]
        output_blocks.append(thick_divider)
        return "\n".join(output_blocks)


# ═══════════════════════════════════════════════════════════════════
# 12. PIPELINE CONTEXT INGESTION RUNTIME
# ═══════════════════════════════════════════════════════════════════
def ingest_repository(
    repo:           dict,
    qdrant_url:     Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    auto_index:     bool          = True,
) -> IngestionResult:
    """Runs a single repository entry metadata payload map through the Osiris pipeline processing layout."""
    repo_id = repo.get("id", f"anonymous/unidentified-node-{uuid.uuid4().hex[:6]}")
    
    # Run Pipeline Stages
    # Run Pipeline Stages
    extracted_tags = extract_tags(repo_id, repo.get("extracted_paragraphs", []))
    documentation_metrics = score_documentation(repo)
    ecosystem_activity = activity_score(repo)
    flux_trends = trend_velocity(repo)
    taxonomy_category = classify_category(repo, extracted_tags)
    
    structured_abstract = build_structured_summary(repo, extracted_tags, taxonomy_category)
    composite_hybrid_vector = generate_hybrid_embedding(repo, extracted_tags, taxonomy_category, structured_abstract)
    
    # Establish persistent engine cluster tracking snapshots before processing decisions
    spatial_topography_data = dynamic_cluster_discovery(_internal_corpus_history)

    # Establish persistent engine cluster tracking snapshots before processing decisions
    spatial_topography_data = dynamic_cluster_discovery(_internal_corpus_history)

    # ── QDRANT ANN SEARCH: Retrieve Top 20 Neighbors ──
    qdrant_hits = []
    corpus_total = len(_internal_corpus_history)
    q_client = None
    
    if _QDRANT_OK:
        try:
            q_client = _resolve_qdrant_client(qdrant_url, qdrant_api_key)
            _verify_qdrant_collection(q_client, composite_hybrid_vector.shape[0])
            
            corpus_total = _qdrant_count_points(q_client)
            
            if corpus_total > 0:
                qdrant_hits = _query_qdrant_neighbors(q_client, composite_hybrid_vector, limit=20)
        except Exception as e:
            print(f"\n[CRITICAL DB ERROR] Qdrant Search Failed: {e}")

    if not qdrant_hits and _internal_corpus_history:
        qdrant_hits = _query_internal_neighbors(composite_hybrid_vector, limit=20)
        corpus_total = max(corpus_total, len(_internal_corpus_history))
        print("[QDRANT WARNING] Falling back to in-memory corpus neighbor scan.")

    # ── INJECTED TELEMETRY DIAGNOSTICS ──
    print(f"\n[DEBUG] Target Repo ID: {repo_id}")
    print(f"[DEBUG] Qdrant corpus_total: {corpus_total}")
    if _QDRANT_OK and q_client is not None:
        print(f"[DEBUG] Qdrant diagnostics: {_qdrant_collection_diagnostics(q_client)}")
    print(f"[DEBUG] Neighbors retrieved: {len(qdrant_hits)}")
    
    neighbor_ids = [hit["repo_id"] for hit in qdrant_hits]
    sim_scores = [round(hit["sim"], 4) for hit in qdrant_hits]
    print(f"[DEBUG] Neighbor IDs: {neighbor_ids}")
    print(f"[DEBUG] Sim Scores: {sim_scores}\n")

    # Execute Multi-Dimensional Metric Computations against the Top 20 slice
    novelty_eval_matrix = compute_multi_dimensional_novelty(
        repo_id=repo_id,
        target_cat=taxonomy_category,
        activity_val=ecosystem_activity,
        target_tags=extracted_tags,
        qdrant_hits=qdrant_hits,     # ◄ Passing the Qdrant Top 20
        corpus_volume=corpus_total
    )

    # ── INGESTION MATRIX DECISION GATEWAY ──
    matrix_quadrant, gate_decision = calculate_quadrant(novelty_eval_matrix.final, flux_trends)
    
    if gate_decision == "REJECTED":
        decision_status = "REJECTED"
        colliding_node_id = novelty_eval_matrix.top_k[0]['repo_id'].split('/')[-1] if novelty_eval_matrix.top_k else '—'
        colliding_node_sim = novelty_eval_matrix.top_k[0]['sim'] if novelty_eval_matrix.top_k else 0.0
        
        if "Copycats" in matrix_quadrant:
            rejection_reason = (
                f"Dropped under Matrix Quadrant: {matrix_quadrant}. "
                f"The repository shows high traction velocity ({flux_trends:.4f}) but low functional novelty ({novelty_eval_matrix.final:.4f}). "
                f"Highly critical correlation drift matching existing node: '{colliding_node_id}' (Similarity={colliding_node_sim:.4f})."
            )
        else:
            rejection_reason = (
                f"Dropped under Matrix Quadrant: {matrix_quadrant}. "
                f"The repository registers cold momentum/velocity values and low novelty parameters ({novelty_eval_matrix.final:.4f})."
            )
    else:
        decision_status = "APPROVED"
        rejection_reason = ""
        
        if auto_index:
            _internal_corpus_history.append({
                "repo_id":  repo_id,
                "vector":   composite_hybrid_vector,
                "category": taxonomy_category,
                "activity": ecosystem_activity,
                "tags":     extracted_tags
            })
            
            _growth_timeline_stream.append({
                "growth_index": len(_internal_corpus_history),
                "repo_id": repo_id,
                "novelty_index_point": novelty_eval_matrix.final
            })
            
            if _QDRANT_OK and q_client is not None:
                try:
                    _index_qdrant_point(q_client, repo_id, composite_hybrid_vector, {
                        "category": taxonomy_category, "activity": ecosystem_activity, "tags": extracted_tags,
                        "doc_quality": documentation_metrics.score, "trend": flux_trends,
                        "quadrant": matrix_quadrant 
                    })
                except Exception:
                    pass

    return IngestionResult(
        repo_id=repo_id, decision=decision_status, rejection_reason=rejection_reason,
        doc_quality=documentation_metrics.score, activity_score=ecosystem_activity, trend_velocity=flux_trends,
        novelty=novelty_eval_matrix, tags=extracted_tags, category=taxonomy_category,
        structured_summary=structured_abstract, doc_found=documentation_metrics.found, doc_missing=documentation_metrics.missing,
        topological_metrics=spatial_topography_data,
        quadrant=matrix_quadrant
    )


# ═══════════════════════════════════════════════════════════════════
# 13. STREAM PIPELINE BATCH EXECUTION & HIGH-LEVEL REPORTING
# ═══════════════════════════════════════════════════════════════════
def ingest_batch(repos: list[dict], verbose: bool = True) -> list[IngestionResult]:
    """Processes sequential stream vectors into the target processing system maps."""
    execution_results = []
    for iteration, repository_map in enumerate(repos):
        result_node = ingest_repository(repository_map)
        if verbose:
            print(result_node)
        execution_results.append(result_node)
    return execution_results

def print_batch_summary(results: list[IngestionResult]) -> None:
    """Generates an extensive, elite terminal dashboard charting structural ecosystem growth profiles."""
    approved_nodes = [node for node in results if node.decision == "APPROVED"]
    rejected_nodes = [node for node in results if node.decision == "REJECTED"]
    novelty_floats = [node.novelty.final for node in results]
    
    width = 76
    double_boundary = "═" * width
    thin_boundary = "─" * width
    
    from collections import Counter
    taxonomy_distribution = Counter(node.category for node in approved_nodes)
    quadrant_distribution = Counter(node.quadrant for node in results)
    
    print(thin_boundary)
    print("  Ecosystem Retrieval 2x2 Matrix Quadrant Breakdown:")
    quadrants_to_check = ["🔥 Viral Rockets", "💎 Hidden Gems", "⚠️ Copycats / Clones", "💤 Dormant Ecosystem Nodes"]
    for quad in quadrants_to_check:
        frequency = quadrant_distribution.get(quad, 0)
        bar_graph_visualization = "▓" * min(frequency, 30)
        print(f"    {quad:<30} [{frequency:>2}]  {bar_graph_visualization}")
    
    highest_novelty_rankings = sorted(approved_nodes, key=lambda x: x.novelty.final, reverse=True)[:5]
    convergent_density_rankings = sorted(approved_nodes, key=lambda x: x.novelty.final)[:5]
    flagged_anomalies = [node for node in results if node.novelty.anomaly_tag != "NONE"]

    print("\n" + double_boundary)
    print(f"               OSIRIS RESEARCH ENGINE STREAM ANALYTICS REPORT")
    print(double_boundary)
    print(f"  Processed Node Evaluation Streams : {len(results)}")
    print(f"  Approved Active Ecosystem Signatures: {len(approved_nodes)}")
    print(f"  Rejected Conflict Vector Drops     : {len(rejected_nodes)}")
    print(f"  Total Flagged System Anomalies    : {len(flagged_anomalies)}")
    
    if novelty_floats:
        print(thin_boundary)
        print(f"  Corpus Ecosystem Novelty Mean Value : {sum(novelty_floats)/len(novelty_floats):.4f}")
        print(f"  Absolute Delta Floor Min Recorded   : {min(novelty_floats):.4f}")
        print(f"  Absolute Delta Peak Max Recorded   : {max(novelty_floats):.4f}")
    print(thin_boundary)
    
    print(f"  Active Taxonomy Spatial Distributions (Registered Approved Nodes):")
    for category_name, frequency in taxonomy_distribution.most_common():
        bar_graph_visualization = "█" * min(frequency, 30)
        print(f"    {category_name:<30} [{frequency:>2}]  {bar_graph_visualization}")
    print(thin_boundary)
    
    print(f"  Top 5 High-Novelty Ecosystem Disruptors:")
    for rank, node in enumerate(highest_novelty_rankings, 1):
        print(f"    {rank}. {node.repo_id:<44} Novelty Vector Index: {node.novelty.final:.4f} [{node.category}]")
    print(thin_boundary)
    
    print(f"  Top 5 Base-Convergent Alternates (Approved Near Minimum Floor Boundaries):")
    for rank, node in enumerate(convergent_density_rankings, 1):
        nearest_match_signature = node.novelty.top_k[0]["repo_id"].split("/")[-1] if node.novelty.top_k else "None (Corpus Origin Seed)"
        print(f"    {rank}. {node.repo_id:<44} Novelty Vector Index: {node.novelty.final:.4f} (Match: {nearest_match_signature})")
        
    if _growth_timeline_stream:
        print(thin_boundary)
        print(f"  Ecosystem Corpus Growth & Novelty Vector Decay Timeline Analysis:")
        sampling_step_interval = max(1, len(_growth_timeline_stream) // 5)
        for position in range(0, len(_growth_timeline_stream), sampling_step_interval):
            timeline_node = _growth_timeline_stream[position]
            print(f"    [Corpus Size Node: {timeline_node['growth_index']:>2}]  Source Module: {timeline_node['repo_id']:<32} Registered Novelty Tracking: {timeline_node['novelty_index_point']:.4f}")
            
    if rejected_nodes:
        print(thin_boundary)
        print(f"  Rejected System Drop Registers ({len(rejected_nodes)} Conflicting Nodes):")
        for node in rejected_nodes:
            print(f"    [!] Drop Track -> {node.repo_id:<42} Matrix Evaluation Value: {node.novelty.final:.4f}")
            
    print(double_boundary + "\n")


# ═══════════════════════════════════════════════════════════════════
# SYSTEM EXECUTION GATEWAY ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    target_data_json_path = "/mnt/user-data/uploads/staged_repositories.json"
    if not os.path.exists(target_data_json_path):
        target_data_json_path = os.path.join(os.path.dirname(__file__), "staged_repositories.json")
        
    if not os.path.exists(target_data_json_path):
        print("⚠️  Context Warning: JSON repository source data profile asset not found. Initializing pipeline standalone test modules...")
        smoke_payload_mock = [
            {"id": "facebook/react",       "star_count": 246000, "pushed_days_ago": 4, 
             "mentionable_users_count": 2, "primary_language": "JavaScript", 
             "readme_length": 5317, "readme_to_codebase_ratio": 0.000005, 
             "extracted_paragraphs": ["React is a JavaScript library for building component based rich interactive user interfaces for web platforms."], 
             "delta_3d": 370, "delta_7d": 1048, "delta_30d": 3499},
            {"id": "vuejs/vue",            "star_count": 208000, "pushed_days_ago": 7, 
             "mentionable_users_count": 3, "primary_language": "JavaScript", 
             "readme_length": 3100, "readme_to_codebase_ratio": 0.000008, 
             "extracted_paragraphs": ["Vue.js is a progressive modular component-based JavaScript framework for engineering clean dynamic user interface layouts."], 
             "delta_3d": 120, "delta_7d": 340, "delta_30d": 1100},
            {"id": "bitcoin/bitcoin",       "star_count": 77000,  "pushed_days_ago": 1, 
             "mentionable_users_count": 5, "primary_language": "C++", 
             "readme_length": 8000, "readme_to_codebase_ratio": 0.00002, 
             "extracted_paragraphs": ["Bitcoin Core is the foundational decentralized open-source p2p cryptocurrency blockchain node implementation engine."], 
             "delta_3d": 40,  "delta_7d": 110, "delta_30d": 380},
            {"id": "anomalous/react-shallow-wrapper", "star_count": 12, "pushed_days_ago": 2, 
             "mentionable_users_count": 1, "primary_language": "JavaScript", 
             "readme_length": 5100, "readme_to_codebase_ratio": 0.000005, 
             "extracted_paragraphs": ["React is a JavaScript library for building component based rich interactive user interfaces for web platforms."], 
             "delta_3d": 2, "delta_7d": 5, "delta_30d": 10}
        ]
        batch_output_nodes = ingest_batch(smoke_payload_mock, verbose=True)
        print_batch_summary(batch_output_nodes)
        sys.exit(0)
        
    with open(target_data_json_path) as data_file_descriptor:
        active_staged_payload = json.load(data_file_descriptor)
        
    print(f"🚀 Osiris Core Framework Booted. Processing comprehensive stream matrix tracking across {len(active_staged_payload)} staged inputs.\n")
    
    # Executing calculation streams directly across ALL 97 entries in data asset arrays
    resolved_batch_matrix = ingest_batch(active_staged_payload, verbose=True)
    print_batch_summary(resolved_batch_matrix)

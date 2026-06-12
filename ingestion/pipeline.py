import uuid
from typing import Optional

from .features import extract_tags, score_documentation, activity_score, trend_velocity, build_structured_summary
from .classification import classify_category
from .embeddings import generate_hybrid_embedding
from .vector_store import _resolve_qdrant_client, _verify_qdrant_collection, _qdrant_count_points, _query_qdrant_neighbors, _query_internal_neighbors, _index_qdrant_point, _qdrant_collection_diagnostics, _QDRANT_OK
from .novelty import compute_multi_dimensional_novelty, calculate_quadrant
from .corpus import CorpusStore, dynamic_cluster_discovery
from .result import IngestionResult

def ingest_repository(
    repo:           dict,
    corpus_store:   'CorpusStore',
    qdrant_url:     Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    auto_index:     bool          = True,
    compute_topology: bool        = False,
) -> IngestionResult:
    """Runs a single repository entry metadata payload map through the Osiris pipeline processing layout."""
    repo_id = repo.get("id", f"anonymous/unidentified-node-{uuid.uuid4().hex[:6]}")
    
    extracted_tags = extract_tags(repo_id, repo.get("extracted_paragraphs", []))
    documentation_metrics = score_documentation(repo)
    ecosystem_activity = activity_score(repo)
    flux_trends = trend_velocity(repo)
    taxonomy_category = classify_category(repo, extracted_tags)
    
    structured_abstract = build_structured_summary(repo, extracted_tags, taxonomy_category)
    composite_hybrid_vector = generate_hybrid_embedding(repo, extracted_tags, taxonomy_category, structured_abstract)
    
    spatial_topography_data = {}
    if compute_topology:
        spatial_topography_data = dynamic_cluster_discovery(corpus_store.get_history())

    qdrant_hits = []
    corpus_total = len(corpus_store.get_history())
    q_client = None
    
    if _QDRANT_OK:
        try:
            q_client = _resolve_qdrant_client(qdrant_url, qdrant_api_key)
            _verify_qdrant_collection(q_client, composite_hybrid_vector.shape[0])
            
            corpus_total = _qdrant_count_points(q_client)
            
            if corpus_total > 0:
                qdrant_hits = _query_qdrant_neighbors(q_client, composite_hybrid_vector, limit=20)
        except Exception as e:
            print(f"\\n[CRITICAL DB ERROR] Qdrant Search Failed: {e}")

    if not qdrant_hits and corpus_store.get_history():
        qdrant_hits = _query_internal_neighbors(composite_hybrid_vector, corpus_store.get_history(), limit=20)
        corpus_total = max(corpus_total, len(corpus_store.get_history()))
        print("[QDRANT WARNING] Falling back to in-memory corpus neighbor scan.")

    # Execute Multi-Dimensional Metric Computations against the Top 20 slice
    novelty_eval_matrix = compute_multi_dimensional_novelty(
        repo_id=repo_id,
        target_cat=taxonomy_category,
        activity_val=ecosystem_activity,
        target_tags=extracted_tags,
        qdrant_hits=qdrant_hits,
        corpus_volume=corpus_total
    )

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
            corpus_store.add_node({
                "repo_id":  repo_id,
                "vector":   composite_hybrid_vector,
                "category": taxonomy_category,
                "activity": ecosystem_activity,
                "tags":     extracted_tags
            }, novelty_eval_matrix.final)
            
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
        embedding=composite_hybrid_vector.tolist(), topological_metrics=spatial_topography_data,
        quadrant=matrix_quadrant
    )


def ingest_batch(repos: list[dict], corpus_store: 'CorpusStore' = None, verbose: bool = True) -> list[IngestionResult]:
    """Processes sequential stream vectors into the target processing system maps."""
    if corpus_store is None:
        corpus_store = CorpusStore()
    execution_results = []
    for iteration, repository_map in enumerate(repos):
        result_node = ingest_repository(repository_map, corpus_store=corpus_store)
        if verbose:
            print(result_node)
        execution_results.append(result_node)
    return execution_results

def print_batch_summary(results: list[IngestionResult], corpus_timeline: list[dict] = None) -> None:
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

    print("\\n" + double_boundary)
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
        
    if corpus_timeline:
        print(thin_boundary)
        print(f"  Ecosystem Corpus Growth & Novelty Vector Decay Timeline Analysis:")
        sampling_step_interval = max(1, len(corpus_timeline) // 5)
        for position in range(0, len(corpus_timeline), sampling_step_interval):
            timeline_node = corpus_timeline[position]
            print(f"    [Corpus Size Node: {timeline_node['growth_index']:>2}]  Source Module: {timeline_node['repo_id']:<32} Registered Novelty Tracking: {timeline_node['novelty_index_point']:.4f}")
            
    if rejected_nodes:
        print(thin_boundary)
        print(f"  Rejected System Drop Registers ({len(rejected_nodes)} Conflicting Nodes):")
        for node in rejected_nodes:
            print(f"    [!] Drop Track -> {node.repo_id:<42} Matrix Evaluation Value: {node.novelty.final:.4f}")
            
    print(double_boundary + "\\n")

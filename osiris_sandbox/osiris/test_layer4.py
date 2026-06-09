# test_layer4.py
import numpy as np
from storage import compute_novelty, store_approved_candidate

def run_sandbox_test():
    print("🚀 Booting Storage & Retrieval Sandbox...\n")

    # 1. Create a fake "dummy" vector (384 dimensions to mimic MiniLM)
    print("[*] Generating simulated 384-dimensional repository vector...")
    dummy_vector = np.random.rand(384).astype(np.float32)

    # 2. Test 1: The Initial Seed (Empty Database)
    print("\n[*] TEST 1: First Repository Ingestion (Should be 1.000 Novelty)")
    result1 = compute_novelty(
        repo_id="test/first-repo",
        target_vector=dummy_vector,
        target_cat="AI Agent",
        activity_val=0.85,
        target_tags=["agent", "llm"]
    )
    print(f"    -> Novelty Score: {result1.final:.4f}")
    print(f"    -> Explanation:\n{result1.explanation}")

    # 3. Save the seed into the database
    print("\n[*] Storing First Repository into Qdrant/Memory...")
    store_approved_candidate(
        repo_id="test/first-repo", 
        vector=dummy_vector, 
        metadata={"category": "AI Agent", "activity": 0.85, "tags": ["agent", "llm"], "novelty_score": result1.final}
    )

    # 4. Test 2: The Copycat (Testing against the seed)
    print("\n[*] TEST 2: Second Repository Ingestion (Testing Retrieval & Math)")
    # We use a slightly modified vector to simulate a near-clone
    clone_vector = dummy_vector * 0.99 
    
    result2 = compute_novelty(
        repo_id="test/second-repo",
        target_vector=clone_vector,
        target_cat="AI Agent",
        activity_val=0.90,
        target_tags=["agent", "llm", "clone"]
    )
    print(f"    -> Novelty Score: {result2.final:.4f}")
    print(f"    -> Matrix Anomaly Flag: {result2.anomaly_tag}")
    print(f"    -> Explanation:\n{result2.explanation}")

if __name__ == "__main__":
    run_sandbox_test()
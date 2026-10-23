\"\"\"Small deterministic metrics to support retrieval experiments.\"\"\"

from collections.abc import Sequence


def recall_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    \"\"\"Compute Recall@k for one query, given relevant and retrieved document IDs.\"\"\"
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)

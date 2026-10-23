from advanced_rag.evaluation.metrics import recall_at_k


def test_recall_at_k_finds_relevant_documents() -> None:
    assert recall_at_k({"a", "b"}, ["x", "a", "b"], 3) == 1.0


def test_recall_at_k_returns_zero_when_none_found() -> None:
    assert recall_at_k({"a"}, ["b", "c"], 2) == 0.0


def test_recall_at_k_handles_no_relevant_documents() -> None:
    assert recall_at_k(set(), ["a"], 1) == 0.0

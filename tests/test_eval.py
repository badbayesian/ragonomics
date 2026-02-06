from ragonometrics.eval import (
    answer_has_citation,
    evaluate_answers,
    evaluate_retrieval,
    mrr_at_k,
    recall_at_k,
)


def test_recall_and_mrr_at_k():
    retrieved = ["1", "2", "3"]
    expected = ["2"]
    assert recall_at_k(retrieved, expected, 1) == 0.0
    assert recall_at_k(retrieved, expected, 2) == 1.0
    assert mrr_at_k(retrieved, expected, 2) == 1.0 / 2


def test_evaluate_retrieval_pages():
    retrieved_meta = [{"page": 1}, {"page": 2}, {"page": 3}]
    metrics = evaluate_retrieval(retrieved_meta, expected_pages=[2], k=2)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr_at_k"] == 0.5


def test_answer_metrics():
    ans = "Finding summary (page 2 words 10-20)"
    assert answer_has_citation(ans)
    metrics = evaluate_answers([ans, "No citation here"])
    assert metrics["citation_coverage"] == 0.5
    assert metrics["hallucination_rate_proxy"] == 0.5



"""Evaluate FinSight retrieval quality."""

import json
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder

from finsight.index_filings import MODEL_NAME
from finsight.retrieval import RERANKER_NAME, retrieve_chunks
from finsight.filing_resolver import resolve_annual_filing


EVAL_PATH = Path(__file__).with_name("eval_cases.json")


def evaluate_case(results, relevant_ids):
    """Calculate Hit@10, MRR@10 and nDCG@10."""

    retrieved_ids = [row[0] for row, score in results[:10]]

    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(retrieved_ids, start=1)
        if chunk_id in relevant_ids
    ]

    hit = 1 if relevant_ranks else 0
    mrr = 1 / relevant_ranks[0] if relevant_ranks else 0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank in relevant_ranks
    )

    ideal_count = min(len(relevant_ids), 10)

    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    ndcg = dcg / ideal_dcg

    return hit, mrr, ndcg


def main():
    """Run evaluation against the indexed SEC documents."""

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty list")

    # Load models once for all questions.
    model = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME)

    scores = []

    for case in cases:
        question = case["question"]
        ticker = case["ticker"]
        relevant_ids = set(case["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError(f"No relevance labels for {case['id']}")

        accession = resolve_annual_filing(question, ticker)

        results = retrieve_chunks(
            question,
            model,
            reranker,
            ticker=ticker,
            accession_number=accession,
        )

        hit, mrr, ndcg = evaluate_case(results, relevant_ids)

        scores.append((hit, mrr, ndcg))

        print(f"\nQuestion: {case['id']}")
        print(f"Hit@10:  {hit}")
        print(f"MRR@10:  {mrr:.4f}")
        print(f"nDCG@10: {ndcg:.4f}")

    count = len(scores)

    print("\n=== Overall Evaluation ===")
    print(f"Questions: {count}")
    print(f"Hit@10:  {sum(s[0] for s in scores) / count:.4f}")
    print(f"MRR@10:  {sum(s[1] for s in scores) / count:.4f}")
    print(f"nDCG@10: {sum(s[2] for s in scores) / count:.4f}")


if __name__ == "__main__":
    main()


"""Retrieve and rerank financial chunks using PostgreSQL and NumPy."""

import numpy as np
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder

from finsight.database import get_connection
from finsight.index_filings import MODEL_NAME
from finsight.filing_resolver import resolve_annual_filing


RERANKER_NAME = "BAAI/bge-reranker-base"


def retrieve_chunks(
    question,
    model,
    reranker,
    ticker="NVDA",
    limit=50,
    accession_number=None,
    section_id=None,
):
    """Retrieve financial chunks using hybrid search and BGE reranking."""

    # 1. Validate inputs.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker cannot be empty")

    question = question.strip()
    ticker = ticker.strip().upper()

    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50")

    if accession_number is not None:
        accession_number = accession_number.strip()

        if not accession_number:
            raise ValueError("Accession number cannot be empty")

    if section_id is not None:
        section_id = section_id.strip()

        if not section_id:
            raise ValueError("Section ID cannot be empty")

    # 2. Generate query embedding.
    query = (
        f"Represent this sentence for searching relevant passages: {question}"
    )

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    if query_embedding.shape != (384,):
        raise ValueError("Expected 384-dimensional query embedding")

    # 3. Retrieve stored embeddings from PostgreSQL.
    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                chunk_id,
                text,
                heading,
                source_url,
                accession_number,
                embedding
            FROM chunks
            WHERE ticker = %s
              AND (%s::text IS NULL OR accession_number = %s)
              AND (%s::text IS NULL OR section_id = %s)
            """,
            (
                ticker,
                accession_number,
                accession_number,
                section_id,
                section_id,
            ),
        ).fetchall()

        if not rows:
            return []

        # 4. PostgreSQL keyword search.
        keyword_results = conn.execute(
            """
            SELECT chunk_id
            FROM chunks
            WHERE ticker = %s
              AND (%s::text IS NULL OR accession_number = %s)
              AND (%s::text IS NULL OR section_id = %s)
              AND to_tsvector('english', text)
                  @@ websearch_to_tsquery('english', %s)
            ORDER BY ts_rank_cd(
                to_tsvector('english', text),
                websearch_to_tsquery('english', %s)
            ) DESC
            LIMIT %s
            """,
            (
                ticker,
                accession_number,
                accession_number,
                section_id,
                section_id,
                question,
                question,
                limit,
            ),
        ).fetchall()

    # 5. Calculate cosine similarity using NumPy.
    document_embeddings = np.asarray(
        [row[5] for row in rows],
        dtype=np.float32,
    )

    if document_embeddings.shape != (len(rows), 384):
        raise ValueError("Invalid stored embedding dimensions")

    if not np.isfinite(document_embeddings).all():
        raise ValueError("Stored embeddings contain invalid values")

    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding contains invalid values")

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
    )

    query_norm = np.linalg.norm(query_embedding)

    if np.any(document_norms == 0) or query_norm == 0:
        raise ValueError("Zero-length embedding found")

    similarities = (
        document_embeddings @ query_embedding
    ) / (document_norms * query_norm)

    similarities = np.clip(similarities, -1.0, 1.0)

    # Convert cosine similarity to cosine distance.
    distances = 1.0 - similarities

    # Preserve the existing result structure.
    all_results = {}

    for row, distance in zip(rows, distances, strict=True):
        all_results[row[0]] = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            float(distance),
        )

    # 6. Select top vector matches.
    top_indices = np.argsort(-similarities)[:limit]

    results = {
        rows[index][0]: all_results[rows[index][0]]
        for index in top_indices
    }

    # 7. Add keyword matches, removing duplicates.
    for (chunk_id,) in keyword_results:
        results[chunk_id] = all_results[chunk_id]

    results = list(results.values())

    if not results:
        return []

    # 8. Include section headings for reranking.
    passages = [
        f"Section: {row[2]}\nPassage: {row[1]}"
        for row in results
    ]

    ranked = reranker.rank(
        question,
        passages,
        top_k=min(10, len(passages)),
        batch_size=4,
    )

    # 9. Return reranked financial evidence.
    final_results = []

    for item in ranked:
        row = results[item["corpus_id"]]
        score = float(item["score"])

        final_results.append((row, score))

    return final_results


if __name__ == "__main__":

    question = input("\nAsk a financial question: ").strip()

    ticker = "NVDA"

    # Automatically select an annual filing when supported.
    try:
        accession = resolve_annual_filing(question, ticker)
    except ValueError as error:
        raise SystemExit(
            f"Filing selection failed: {error}"
        ) from error

    if accession:
        print(f"\nAutomatically selected filing: {accession}")
    else:
        print("\nNo fiscal year specified. Searching all indexed filings.")

    # Load models using the available GPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nModel device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    reranker = CrossEncoder(
        RERANKER_NAME,
        device=device,
    )

    # Retrieve financial evidence.
    results = retrieve_chunks(
        question,
        model,
        reranker,
        ticker=ticker,
        accession_number=accession,
    )

    if not results:
        print("No matching chunks found")

    else:
        for index, (row, score) in enumerate(results, start=1):

            print(f"\nResult: {index}")
            print(f"Chunk ID: {row[0]}")
            print(f"Text: {row[1]}")
            print(f"Section: {row[2]}")
            print(f"Source: {row[3]}")
            print(f"Accession: {row[4]}")
            print(f"Cosine Distance: {row[5]:.4f}")
            print(f"Rerank Score: {score:.4f}")

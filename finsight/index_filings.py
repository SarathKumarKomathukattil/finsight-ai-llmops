
"""Index processed SEC filings in PostgreSQL."""
import torch
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from finsight.chunking import chunk_filing
from finsight.database import get_connection
from finsight.filing_parser import PARSER_VERSION


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    """Index SEC filings, metadata, chunks, and embeddings."""

    folder = PROJECT_ROOT / "data" / "processed" / "sec"

    files = sorted(folder.rglob(f"*.{PARSER_VERSION}.json"))

    if not files:
        raise FileNotFoundError("No processed filings found")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    for file in files:

        # 1. Read processed SEC filing.
        data = json.loads(file.read_text(encoding="utf-8"))

        # 2. Generate financial document chunks.
        chunks = chunk_filing(data)

        texts = [chunk["text"] for chunk in chunks]

        # 3. Generate embeddings.
        embeddings = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        if embeddings.shape[1] != 384:
            raise ValueError("Expected 384-dimensional embeddings")

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk["embedding"] = embedding.tolist()

        # 4. Save filing metadata and chunks in one transaction.
        with get_connection() as conn:

            # 4A. Insert or update filing metadata.
            conn.execute(
                """
                INSERT INTO filings (
                    accession_number,
                    ticker,
                    filing_type,
                    filing_date,
                    report_date,
                    source_url,
                    source_sha256
                )
                VALUES (
                    %(accession_number)s,
                    %(ticker)s,
                    %(filing_type)s,
                    %(filing_date)s,
                    %(report_date)s,
                    %(source_url)s,
                    %(source_sha256)s
                )
                ON CONFLICT (accession_number)
                DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    filing_type = EXCLUDED.filing_type,
                    filing_date = EXCLUDED.filing_date,
                    report_date = COALESCE(
                        EXCLUDED.report_date,
                        filings.report_date
                    ),
                    source_url = EXCLUDED.source_url,
                    source_sha256 = EXCLUDED.source_sha256
                """,
                {
                    "accession_number": data["accession_number"],
                    "ticker": data["ticker"],
                    "filing_type": data["filing_type"],
                    "filing_date": data["filing_date"],
                    "report_date": data.get("report_date") or None,
                    "source_url": data["source"]["url"],
                    "source_sha256": data["source"]["sha256"],
                },
            )

            # 4B. Remove old chunks for this filing.
            conn.execute(
                """
                DELETE FROM chunks
                WHERE accession_number = %s
                """,
                (data["accession_number"],),
            )

            # 4C. Insert updated chunks and embeddings.
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunks (
                        chunk_id,
                        ticker,
                        filing_type,
                        filing_date,
                        accession_number,
                        section_id,
                        heading,
                        chunk_index,
                        text,
                        source_url,
                        source_sha256,
                        embedding
                    )
                    VALUES (
                        %(chunk_id)s,
                        %(ticker)s,
                        %(filing_type)s,
                        %(filing_date)s,
                        %(accession_number)s,
                        %(section_id)s,
                        %(heading)s,
                        %(chunk_index)s,
                        %(text)s,
                        %(source_url)s,
                        %(source_sha256)s,
                        %(embedding)s
                    )
                    """,
                    chunks,
                )

        print(
            f"{data['ticker']} | "
            f"{data['filing_type']} | "
            f"{len(chunks)} chunks indexed"
        )


if __name__ == "__main__":
    main()


from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def chunk_filing(data: dict) -> list[dict]:
    """Create text chunks from financial document sections."""

    sections = data["sections"]

    if not sections:
        raise ValueError("No narrative sections found")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            r"(?<=[.!?])\s+",
            " ",
            "",
        ],
        is_separator_regex=True,
        keep_separator="end",
    )

    chunks = []

    for section in sections:
        text = section["text"]

        if not text.strip():
            continue

        pieces = splitter.split_text(text)

        for index, piece in enumerate(pieces):

            chunk_id = (
                f"{data['accession_number']}_"
                f"{section['section_id']}_{index}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "ticker": data["ticker"],
                "filing_type": data["filing_type"],
                "filing_date": data["filing_date"],
                "accession_number": data["accession_number"],
                "section_id": section["section_id"],
                "heading": section["heading"],
                "chunk_index": index,
                "text": piece,
                "source_url": data["source"]["url"],
                "source_sha256": data["source"]["sha256"],
            })

    if not chunks:
        raise ValueError("No usable chunks generated")

    return chunks

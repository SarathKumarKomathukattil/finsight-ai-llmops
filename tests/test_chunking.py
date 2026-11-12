import pytest

from finsight.chunking import chunk_filing, CHUNK_SIZE


@pytest.fixture
def sample_filing():
    return {
        "ticker": "NVDA",
        "filing_type": "10-Q",
        "filing_date": "2026-08-26",
        "accession_number": "0001045810-26-000075",
        "source": {
            "url": "https://www.sec.gov/example.htm",
            "sha256": "a" * 64,
        },
        "sections": [
            {
                "section_id": "part_i_item_2",
                "heading": "Management Discussion",
                "text": "Revenue increased significantly. " * 150
            },
            {
                "section_id": "part_ii_item_1a",
                "heading": "Risk Factors",
                "text": "Supply chain risks may affect operations. " * 150
            }
        ]
    }


def test_chunk_filing(sample_filing):
    chunks = chunk_filing(sample_filing)

    assert len(chunks) > 2

    assert all(len(chunk["text"]) <= CHUNK_SIZE for chunk in chunks)

    assert all(chunk["ticker"] == "NVDA" for chunk in chunks)

    assert all(chunk["source_url"] for chunk in chunks)

    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_repeatable_chunk_ids(sample_filing):
    first = chunk_filing(sample_filing)
    second = chunk_filing(sample_filing)

    assert first == second


def test_empty_sections(sample_filing):
    sample_filing["sections"] = []

    with pytest.raises(ValueError, match="No narrative sections"):
        chunk_filing(sample_filing)
import hashlib

import pytest

from finsight.sec_client import build_filing_url
from finsight.storage import SnapshotStore
from finsight.ingestion import download_latest_10q


def test_build_filing_url():
    url = build_filing_url(
        "0001045810",
        "0001045810-26-000075",
        "nvda-20260726.htm",
    )

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/"
        "1045810/000104581026000075/nvda-20260726.htm"
    )


def test_reject_unsafe_document():
    with pytest.raises(ValueError):
        build_filing_url(
            "0001045810",
            "0001045810-26-000075",
            "../unsafe.htm",
        )


def test_download_latest_10q(tmp_path):
    content = b"<html><body>" + b"Filing content " * 100 + b"</body></html>"

    class FakeClient:
        def get_filing(self, cik, accession, document):
            return content

    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K"],
                "accessionNumber": [
                    "0001045810-26-000075",
                    "0001045810-26-000076",
                ],
                "primaryDocument": [
                    "nvda-20260726.htm",
                    "other.htm",
                ],
                "filingDate": [
                    "2026-08-26",
                    "2026-08-27",
                ],
            }
        }
    }

    store = SnapshotStore(tmp_path)
    client = FakeClient()

    first = download_latest_10q(
        client, store, "0001045810", submissions
    )

    second = download_latest_10q(
        client, store, "0001045810", submissions
    )

    saved_path = tmp_path / first["path"]

    assert saved_path.exists()
    assert saved_path.read_bytes() == content

    assert first["sha256"] == hashlib.sha256(content).hexdigest()
    assert first["path"] == second["path"]
    assert first["form"] == "10-Q"


def test_detect_corrupted_filing(tmp_path):
    store = SnapshotStore(tmp_path)

    content = b"<html>" + b"Original content " * 100 + b"</html>"

    snapshot = store.save_snapshot(
        "0001045810",
        "filings",
        content,
        True,
        accession_number="0001045810-26-000075",
    )

    path = tmp_path / snapshot["path"]
    path.write_bytes(b"modified content")

    with pytest.raises(OSError, match="Corrupted snapshot"):
        store.save_snapshot(
            "0001045810",
            "filings",
            content,
            True,
            accession_number="0001045810-26-000075",
        )
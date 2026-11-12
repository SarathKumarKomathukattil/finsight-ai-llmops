import hashlib
import pytest

from finsight.sec_client import build_filing_url
from finsight.storage import SnapshotStore
from finsight.ingestion import download_latest_10q
from finsight.ingestion import select_filings, download_filing


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
                "reportDate": [
                    "2026-07-26",
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

def test_select_multiple_filings():

    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-K", "10-Q", "8-K"],
                "accessionNumber": [
                    "001", "002", "003", "004", "005"
                ],
                "filingDate": [
                    "2026-08-26",
                    "2026-09-10",
                    "2026-02-25",
                    "2026-05-20",
                    "2026-07-10",
                ],
                "primaryDocument": [
                    "q1.htm",
                    "event1.htm",
                    "annual.htm",
                    "q2.htm",
                    "event2.htm",
                ],
                "reportDate": [
                    "2026-07-26",
                    "2026-09-10",
                    "2026-01-25",
                    "2026-04-26",
                    "2026-07-10",
                ],
            }
        }
    }

    limits = {
        "10-K": 1,
        "10-Q": 2,
        "8-K": 1,
    }

    selected = select_filings(submissions, limits)

    assert len(selected) == 4

    assert [f["form"] for f in selected] == [
        "10-K",
        "10-Q",
        "10-Q",
        "8-K",
    ]

    assert selected[1]["primary_document"] == "q1.htm"
    assert selected[3]["primary_document"] == "event1.htm"

def test_download_multiple_forms(tmp_path):

    html = (
        b"<html><body>"
        + b"Financial disclosure " * 100
        + b"</body></html>"
    )

    class FakeClient:

        def get_filing(self, cik, accession, document):
            return html

    client = FakeClient()

    store = SnapshotStore(tmp_path)

    filings = [
        {
            "form": "10-K",
            "accession_number": "0001045810-26-000001",
            "filing_date": "2026-02-25",
            "primary_document": "annual.htm",
        },
        {
            "form": "8-K",
            "accession_number": "0001045810-26-000002",
            "filing_date": "2026-07-10",
            "primary_document": "event.htm",
        },
    ]

    for filing in filings:

        saved = download_filing(
            client,
            store,
            "0001045810",
            filing,
        )

        path = tmp_path / saved["path"]

        assert path.exists()
        assert path.read_bytes() == html
        assert saved["form"] == filing["form"]
        assert saved["accession_number"] == filing["accession_number"]

    import pytest


def test_invalid_filing_limits():

    submissions = {
        "filings": {
            "recent": {
                "form": [],
                "accessionNumber": [],
                "filingDate": [],
                "primaryDocument": [],
                "reportDate": [],
            }
        }
    }

    with pytest.raises(ValueError, match="Unsupported filing form"):
        select_filings(submissions, {"10-X": 1})

    with pytest.raises(ValueError, match="Invalid filing limit"):
        select_filings(submissions, {"10-K": -1})

import json
from unittest.mock import Mock

import pytest

import finsight.ingestion as ingestion
from finsight.storage import SnapshotStore

CIK = "0001045810"


def make_client(accessions):
    """Create a fake SEC client with sample filings."""

    client = Mock()

    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "form": ["10-Q"] * len(accessions),
                "filingDate": ["2026-10-26"] * len(accessions),
            }
        }
    }

    client.get_submissions.return_value = json.dumps(submissions).encode("utf-8")

    client.get_companyfacts.return_value = b'{"facts": {}}'

    return client


def mock_validators(monkeypatch):
    """Replace real SEC validation for isolated workflow tests."""

    monkeypatch.setattr(
        ingestion,
        "validate_submissions",
        lambda data, cik: None,
    )

    monkeypatch.setattr(
        ingestion,
        "validate_companyfacts",
        lambda data, cik: None,
    )


def test_first_run_and_no_changes(tmp_path, monkeypatch):
    mock_validators(monkeypatch)

    store = SnapshotStore(tmp_path)
    client = make_client(["FILING_001", "FILING_002"])

    # First run: establish baseline
    first = ingestion.ingest_company(client, store, "NVDA", CIK)

    assert first["status"] == "validated"
    assert first["initial_baseline"] is True
    assert first["new_filings"] == []

    state = store.load_state(CIK)

    assert state["accessions"] == ["FILING_001", "FILING_002"]

    # Second run: same filings
    second = ingestion.ingest_company(client, store, "NVDA", CIK)

    assert second["initial_baseline"] is False
    assert second["new_filings"] == []
    assert second["companyfacts_refreshed"] is False

    # Company Facts was needed only during the first run
    assert client.get_companyfacts.call_count == 1


def test_new_filing_and_refresh(tmp_path, monkeypatch):
    mock_validators(monkeypatch)

    store = SnapshotStore(tmp_path)
    client = make_client(["FILING_001"])

    # Establish the initial checkpoint
    ingestion.ingest_company(client, store, "NVDA", CIK)

    # Simulate SEC publishing a new filing
    updated = {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "FILING_001",
                    "FILING_002",
                ],
                "form": ["10-Q", "8-K"],
                "filingDate": [
                    "2026-10-20",
                    "2026-10-26",
                ],
            }
        }
    }

    client.get_submissions.return_value = json.dumps(updated).encode("utf-8")

    result = ingestion.ingest_company(client, store, "NVDA", CIK)

    assert len(result["new_filings"]) == 1
    assert result["new_filings"][0]["accession_number"] == ("FILING_002")
    assert result["new_filings"][0]["form"] == "8-K"
    assert result["companyfacts_refreshed"] is True

    # The new filing should be saved in the checkpoint
    assert "FILING_002" in store.load_state(CIK)["accessions"]

    # Check forced Company Facts refresh
    forced = ingestion.ingest_company(
        client,
        store,
        "NVDA",
        CIK,
        refresh_facts=True,
    )

    assert forced["new_filings"] == []
    assert forced["companyfacts_refreshed"] is True
    assert client.get_companyfacts.call_count == 3


def test_invalid_data_preserves_checkpoint(tmp_path, monkeypatch):
    mock_validators(monkeypatch)

    store = SnapshotStore(tmp_path)
    client = make_client(["FILING_001"])

    # Successful ingestion
    ingestion.ingest_company(client, store, "NVDA", CIK)

    original_state = store.load_state(CIK)

    # Simulate an invalid SEC response
    client.get_submissions.return_value = b"{invalid json"

    with pytest.raises(json.JSONDecodeError):
        ingestion.ingest_company(client, store, "NVDA", CIK)

    # Previous successful checkpoint must remain unchanged
    assert store.load_state(CIK) == original_state

    # Invalid data must be quarantined
    quarantine = list((tmp_path / "quarantine").rglob("*.bin"))

    assert len(quarantine) == 1

    # Failed run should be recorded
    manifests = list((tmp_path / "manifests").rglob("*.json"))

    statuses = [
        json.loads(path.read_text(encoding="utf-8"))["status"] for path in manifests
    ]

    assert "failed" in statuses


def test_companyfacts_failure_preserves_state(tmp_path, monkeypatch):
    mock_validators(monkeypatch)

    store = SnapshotStore(tmp_path)
    client = make_client(["FILING_001"])

    ingestion.ingest_company(client, store, "NVDA", CIK)

    original_state = store.load_state(CIK)

    # Simulate a new filing
    client.get_submissions.return_value = json.dumps(
        {
            "filings": {
                "recent": {
                    "accessionNumber": ["FILING_001", "FILING_002"],
                    "form": ["10-Q", "8-K"],
                    "filingDate": ["2026-10-20", "2026-10-26"],
                }
            }
        }
    ).encode("utf-8")

    # But Company Facts API fails
    client.get_companyfacts.side_effect = RuntimeError("SEC temporarily unavailable")

    with pytest.raises(RuntimeError, match="SEC temporarily"):
        ingestion.ingest_company(client, store, "NVDA", CIK)

    # We must not mark the new filing as processed
    assert store.load_state(CIK) == original_state

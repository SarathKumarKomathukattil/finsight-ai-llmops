import hashlib
import json
import pytest
from pathlib import Path
from finsight.storage import SnapshotStore

CIK = "0001045810"


def test_save_snapshot(tmp_path):
    store = SnapshotStore(tmp_path)
    content = b'{"company": "NVIDIA"}'

    result = store.save_snapshot(CIK, "submissions", content, True)

    path = tmp_path / result["path"]

    assert path.exists()
    assert path.read_bytes() == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()

    # Saving the same content again should not create another file
    second = store.save_snapshot(CIK, "submissions", content, True)

    assert second == result
    assert len(list(path.parent.glob("*.json"))) == 1


def test_quarantine_and_corruption(tmp_path):
    store = SnapshotStore(tmp_path)

    # Invalid data should be stored in quarantine
    result = store.save_snapshot(CIK, "submissions", b"invalid data", False)

    assert Path(result["path"]).parts[0] == "quarantine"
    assert (tmp_path / result["path"]).exists()

    # Detect corruption of an existing snapshot
    content = b'{"test": true}'

    saved = store.save_snapshot(CIK, "submissions", content, True)

    path = tmp_path / saved["path"]
    path.write_bytes(b"corrupted")

    with pytest.raises(OSError, match="Corrupted snapshot"):
        store.save_snapshot(CIK, "submissions", content, True)


def test_state_and_manifest(tmp_path):
    store = SnapshotStore(tmp_path)

    # No checkpoint on the first run
    assert store.load_state(CIK) is None

    state = {"accessions": ["FILING_001", "FILING_002"]}

    store.save_state(CIK, state)

    assert store.load_state(CIK) == state

    # Manifest should record the ingestion result
    store.save_manifest(CIK, {"ticker": "NVDA", "status": "validated"})

    files = list((tmp_path / "manifests").rglob("*.json"))

    assert len(files) == 1

    manifest = json.loads(files[0].read_text(encoding="utf-8"))

    assert manifest["ticker"] == "NVDA"
    assert manifest["status"] == "validated"


def test_invalid_checkpoint(tmp_path):
    store = SnapshotStore(tmp_path)

    path = tmp_path / "state" / "sec" / f"{CIK}.json"
    path.parent.mkdir(parents=True)

    path.write_text('{"accessions": [123]}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid checkpoint"):
        store.load_state(CIK)

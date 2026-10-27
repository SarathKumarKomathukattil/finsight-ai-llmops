import os
import re
import json
import hashlib
from uuid import uuid4
from pathlib import Path
from datetime import datetime,timezone
from finsight.validation import normalize_cik


def write_atomic(path:Path,content:bytes) -> None:
    """Safely replace a file using a temporary file"""

    path.parent.mkdir(parents=True,exist_ok=True)

    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temp.open('wb') as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


class SnapshotStore:
    """Store SEC snapshots, checkpoints and manifests."""

    def __init__(self,root:Path):
        self.root = Path(root)

    def save_snapshot(self,cik: str,kind: str,content: bytes,valid: bool,accession_number: str | None = None,) -> dict:

        if kind == "company_tickers":
            if cik != "_directory":
                raise ValueError("Invalid company directory identifier")

        elif kind in {"submissions", "companyfacts", "filings"}:
            cik = normalize_cik(cik)

        else:
            raise ValueError(f"Invalid SEC dataset: {kind}")

        # Filing documents need their accession number.
        if kind == "filings":
            if not isinstance(accession_number, str) or not re.fullmatch(
                r"\d{10}-\d{2}-\d{6}", accession_number
            ):
                raise ValueError("Invalid filing accession number")

            location = Path("filings") / accession_number

        else:
            if accession_number is not None:
                raise ValueError("Accession number is only valid for filings")

            location = Path(kind)

        digest = hashlib.sha256(content).hexdigest()

        folder = "raw" if valid else "quarantine"

        if not valid:
            extension = "bin"
        elif kind == "filings":
            extension = "html"
        else:
            extension = "json"

        path = (
            self.root
            / folder
            / "sec"
            / cik
            / location
            / f"{digest}.{extension}"
        )

        if path.exists():
            existing = hashlib.sha256(path.read_bytes()).hexdigest()

            if existing != digest:
                raise OSError(f"Corrupted snapshot: {path}")

        else:
            write_atomic(path, content)

        return {
            "path": str(path.relative_to(self.root)),
            "sha256": digest,
        }
    
    def _state_path(self,cik:str) -> Path:
        return self.root/'state'/'sec'/f'{normalize_cik(cik)}.json'

    def load_state(self,cik:str) -> dict | None:
        """Load the previous successful checkpoint."""

        path = self._state_path(cik)
        if not path.exists():
            return None
        state = json.loads(path.read_text(encoding="utf-8"))

        if (
            not isinstance(state, dict)
            or not isinstance(state.get("accessions"), list)
            or not all(
                isinstance(x, str) for x in state["accessions"]
            )
        ):
            raise ValueError(f"Invalid checkpoint: {path}")

        return state

    def save_state(self, cik: str, state: dict) -> None:
        """Save the latest successful checkpoint."""

        content = json.dumps(state, indent=2).encode("utf-8")
        write_atomic(self._state_path(cik), content)

    def save_manifest(self, cik: str, manifest: dict) -> None:
        """Record the details of an ingestion run."""

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        path = (
            self.root / "manifests" / "sec"
            / normalize_cik(cik)
            / f"{timestamp}-{uuid4().hex[:8]}.json"
        )

        content = json.dumps(manifest, indent=2).encode("utf-8")
        write_atomic(path, content)


    






            




    

        






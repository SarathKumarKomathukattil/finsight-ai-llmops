"""Ingest SEC data, save snapshots, and detect new filings."""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from finsight.company_resolver import resolve_ticker
from finsight.sec_client import SECClient, BASE_URL, build_filing_url
from finsight.storage import SnapshotStore
from finsight.validation import (
    normalize_cik,
    validate_submissions,
    validate_companyfacts,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def validate_and_store(store, cik, kind, content, validator):
    """Validate a SEC response and save its original bytes."""

    try:
        data = json.loads(content)
        validator(data, cik)

    except (ValueError, TypeError, KeyError):
        store.save_snapshot(cik, kind, content, False)
        raise

    snapshot = store.save_snapshot(cik, kind, content, True)

    return data, snapshot

def download_latest_10q(client: SECClient,store: SnapshotStore,cik: str,submissions: dict,) -> dict:
        """Download, validate and preserve the latest 10-Q."""

        recent = submissions["filings"]["recent"]

        matches = [
            i for i, form in enumerate(recent["form"])
            if form == "10-Q"
        ]

        if not matches:
            raise ValueError("No 10-Q found in recent SEC filings")

        # Choose the latest filing date.
        i = max(matches, key=lambda index: recent["filingDate"][index])

        accession = recent["accessionNumber"][i]
        document = recent["primaryDocument"][i]
        filing_date = recent["filingDate"][i]

        url = build_filing_url(cik, accession, document)

        content = client.get_filing(cik, accession, document)

        # Reject obviously invalid or incomplete HTML responses.
        if len(content) < 1024 or b"<html" not in content[:8192].lower():
            store.save_snapshot(
                cik,
                "filings",
                content,
                False,
                accession_number=accession,
            )
            raise ValueError("Invalid SEC filing HTML response")

        snapshot = store.save_snapshot(
            cik,
            "filings",
            content,
            True,
            accession_number=accession,
        )

        return {
            "url": url,
            "form": "10-Q",
            "accession_number": accession,
            "primary_document": document,
            "filing_date": filing_date,
            **snapshot,
        }


def ingest_company(client: SECClient,store: SnapshotStore,ticker: str,cik: str,refresh_facts: bool = False,download_10q: bool = False,) -> dict:
    """Download company data and detect newly observed filings."""

    cik = normalize_cik(cik)

    manifest = {
        "ticker": ticker,
        "cik": cik,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "sources": {},
    }

    try:
        # 1. Load previous checkpoint
        previous = store.load_state(cik)

        # 2. Download and validate SEC submissions
        submissions_url = f"{BASE_URL}/submissions/CIK{cik}.json"

        manifest["sources"]["submissions"] = {"url": submissions_url}

        submissions, snapshot = validate_and_store(
            store,
            cik,
            "submissions",
            client.get_submissions(cik),
            validate_submissions,
        )

        manifest["sources"]["submissions"].update(snapshot)

        # 3. Compare current and previously seen filings
        recent = submissions["filings"]["recent"]
        accessions = recent["accessionNumber"]

        known = set(previous["accessions"]) if previous is not None else set()

        new_filings = []

        if previous is not None:
            for i, accession in enumerate(accessions):
                if accession not in known:
                    new_filings.append(
                        {
                            "accession_number": accession,
                            "form": recent["form"][i],
                            "filing_date": recent["filingDate"][i],
                        }
                    )

        # 4. Decide whether Company Facts needs refreshing
        refresh = previous is None or bool(new_filings) or refresh_facts

        facts_hash = previous.get("companyfacts_hash") if previous is not None else None

        if refresh:
            facts_url = f"{BASE_URL}/api/xbrl/" f"companyfacts/CIK{cik}.json"

            manifest["sources"]["companyfacts"] = {"url": facts_url}

            _, facts_snapshot = validate_and_store(
                store,
                cik,
                "companyfacts",
                client.get_companyfacts(cik),
                validate_companyfacts,
            )

            manifest["sources"]["companyfacts"].update(facts_snapshot)

            facts_hash = facts_snapshot["sha256"]

        if download_10q:
            manifest["sources"]["latest_10q"] = download_latest_10q(
                client,
                store,
                cik,
                submissions,
            )

        # 5. Prepare the updated checkpoint
        state = {
            "accessions": sorted(known | set(accessions)),
            "submissions_hash": snapshot["sha256"],
            "companyfacts_hash": facts_hash,
            "last_success_at_utc": (datetime.now(timezone.utc).isoformat()),
        }

        # 6. Record the ingestion result
        manifest.update(
            {
                "status": "validated",
                "initial_baseline": previous is None,
                "new_filings": new_filings,
                "companyfacts_refreshed": refresh,
            }
        )

        # Save the updated checkpoint
        store.save_manifest(cik, manifest)

        store.save_state(cik, state)

        return manifest

    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"

        store.save_manifest(cik, manifest)
        raise


def main():
    """Run SEC ingestion for configured companies."""

    parser = argparse.ArgumentParser(description="Ingest SEC company data")

    parser.add_argument(
        "--refresh-facts",
        action="store_true",
    )

    parser.add_argument(
    "--download-10q",
    action="store_true",
    help="Download the latest 10-Q for each configured company",)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # 1. Read configured tickers
    config_path = PROJECT_ROOT / "config" / "companies.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))

    tickers = config.get("tickers") if isinstance(config, dict) else None

    if (
        not isinstance(tickers, list)
        or not tickers
        or any(not isinstance(t, str) or not t.strip() for t in tickers)
    ):
        raise ValueError("Invalid companies.json configuration")

    # Normalize and remove duplicate tickers
    tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))

    # 2. Initialize storage and SEC client
    store = SnapshotStore(PROJECT_ROOT / "data")
    client = SECClient()

    failures = []

    try:
        # 3. Fetch the official SEC company directory
        raw = client.get_company_directory()

        try:
            directory = json.loads(raw)

            if not isinstance(directory, dict) or not directory:
                raise ValueError("Invalid SEC company directory")

        except (ValueError, TypeError):
            store.save_snapshot(
                "_directory",
                "company_tickers",
                raw,
                False,
            )
            raise

        store.save_snapshot(
            "_directory",
            "company_tickers",
            raw,
            True,
        )

        # 4. Process each configured company
        for ticker in tickers:
            try:
                cik = resolve_ticker(directory, ticker)

                result = ingest_company(
                    client,
                    store,
                    ticker,
                    cik,
                    refresh_facts=args.refresh_facts,
                    download_10q=args.download_10q,
                )

                print(f"{ticker}: " f"{len(result['new_filings'])} new filings")

                if args.download_10q:
                    filing = result["sources"]["latest_10q"]

                    print(
                        f"{ticker}: saved 10-Q "
                        f"{filing['accession_number']} "
                        f"to {filing['path']}")

            except Exception:
                failures.append(ticker)
                logger.exception("Failed to ingest %s", ticker)

    finally:
        client.close()

    if failures:
        raise RuntimeError(f"Failed companies: {', '.join(failures)}")


if __name__ == "__main__":
    main()

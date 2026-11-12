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

def select_filings(submissions: dict, limits: dict) -> list[dict]:
    """Select recent SEC filings according to configured form limits."""

    recent = submissions["filings"]["recent"]

    required = (
    "form",
    "accessionNumber",
    "filingDate",
    "reportDate",
    "primaryDocument",
)

    # SEC stores filing information in parallel arrays.
    lengths = {len(recent[key]) for key in required}

    if len(lengths) != 1:
        raise ValueError("Inconsistent SEC filing metadata")

    selected = []

    for form, limit in limits.items():

        if form not in {"10-K", "10-Q", "8-K"}:
            raise ValueError(f"Unsupported filing form: {form}")

        if type(limit) is not int or limit < 0 or limit > 20:
            raise ValueError(f"Invalid filing limit for {form}")

        matches = []

        for i, filing_form in enumerate(recent["form"]):

            if filing_form == form:
                matches.append({
                    "form": form,
                    "accession_number": recent["accessionNumber"][i],
                    "filing_date": recent["filingDate"][i],
                    "report_date": recent["reportDate"][i] or None,
                    "primary_document": recent["primaryDocument"][i],
                })

        # SEC filing dates use YYYY-MM-DD, so string sorting works.
        matches.sort(
            key=lambda filing: (
                filing["filing_date"],
                filing["accession_number"],
            ),
            reverse=True,
        )

        chosen = matches[:limit]

        if len(chosen) < limit:
            logger.warning(
                "%s: requested %s filings, found %s",
                form,
                limit,
                len(chosen),
            )

        selected.extend(chosen)

    if not selected:
        raise ValueError("No matching SEC filings selected")

    return selected

def download_filing(
    client: SECClient,
    store: SnapshotStore,
    cik: str,
    filing: dict,
) -> dict:
    """Download and preserve one original SEC filing."""

    accession = filing["accession_number"]
    document = filing["primary_document"]

    url = build_filing_url(cik, accession, document)

    content = client.get_filing(cik, accession, document)

    # Basic HTML sanity check.
    valid = (
        len(content) >= 1024
        and b"<html" in content[:8192].lower()
    )

    snapshot = store.save_snapshot(
        cik,
        "filings",
        content,
        valid,
        accession_number=accession,
    )

    if not valid:
        raise ValueError(
            f"Invalid HTML for SEC filing {accession}"
        )

    return {
        **filing,
        "url": url,
        **snapshot,
    }

def download_latest_10q(
    client: SECClient,
    store: SnapshotStore,
    cik: str,
    submissions: dict,
) -> dict:
    """Preserve the existing latest-10-Q functionality."""

    filing = select_filings(
        submissions,
        {"10-Q": 1},
    )[0]

    return download_filing(client, store, cik, filing)

def ingest_company(client: SECClient,store: SnapshotStore,ticker: str,cik: str,refresh_facts: bool = False,download_10q: bool = False,filing_limits: dict | None = None) -> dict:
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


        # Download configured 10-K, 10-Q, and 8-K documents.
        if filing_limits is not None:

            selected = select_filings(submissions, filing_limits)

            manifest["sources"]["filings"] = []

            for filing in selected:

                saved = download_filing(
                    client,
                    store,
                    cik,
                    filing,
                )

                manifest["sources"]["filings"].append(saved)
        

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

    parser.add_argument(
    "--download-filings",
    action="store_true",
    help="Download configured 10-K, 10-Q and 8-K filings",)

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

    # Prevent both download options from running together.
    if args.download_filings and args.download_10q:
        parser.error("Choose either --download-filings or --download-10q")

    filing_limits = None

    if args.download_filings:

        filing_limits = config.get("filings")

        if not isinstance(filing_limits, dict) or not filing_limits:
            raise ValueError("Invalid filing configuration")

        supported_forms = {"10-K", "10-Q", "8-K"}

        for form, limit in filing_limits.items():

            if form not in supported_forms:
                raise ValueError(f"Unsupported filing form: {form}")

            if type(limit) is not int or not 0 <= limit <= 20:
                raise ValueError(f"Invalid filing limit for {form}")

        if not any(filing_limits.values()):
            raise ValueError("At least one filing limit must be positive")
    

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
                    filing_limits=filing_limits,
                )

                print(f"{ticker}: " f"{len(result['new_filings'])} new filings")

                if args.download_10q:
                    filing = result["sources"]["latest_10q"]

                    print(
                        f"{ticker}: saved 10-Q "
                        f"{filing['accession_number']} "
                        f"to {filing['path']}")

                if args.download_filings:

                    filings = result["sources"]["filings"]

                    print(f"{ticker}: {len(filings)} filings downloaded")

                    for filing in filings:

                        print(
                            f"{filing['form']} | "
                            f"{filing['filing_date']} | "
                            f"{filing['accession_number']}"
                        )

            except Exception:
                failures.append(ticker)
                logger.exception("Failed to ingest %s", ticker)

    finally:
        client.close()

    if failures:
        raise RuntimeError(f"Failed companies: {', '.join(failures)}")


if __name__ == "__main__":
    main()

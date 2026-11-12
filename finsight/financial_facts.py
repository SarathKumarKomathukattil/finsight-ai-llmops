
"""Extract annual financial comparisons from SEC Company Facts."""

import argparse
import hashlib
import json

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from finsight.database import get_connection
from finsight.filing_resolver import resolve_annual_filing
from finsight.sec_client import SECClient
from finsight.storage import write_atomic
from finsight.validation import normalize_cik


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use one consistent accounting concept for both years.
METRIC_TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
}


def _annual_pair(companyfacts, accession, report_date, tags):
    """Find comparable full-year USD facts from the same 10-K."""

    gaap = companyfacts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        records = gaap.get(tag, {}).get("units", {}).get("USD", [])

        current = []
        previous = []

        for record in records:
            if not isinstance(record, dict):
                continue

            if (
                record.get("form") != "10-K"
                or record.get("accn") != accession
            ):
                continue

            value = record.get("val")

            if type(value) is not int:
                continue

            try:
                start = date.fromisoformat(record["start"])
                end = date.fromisoformat(record["end"])
            except (KeyError, TypeError, ValueError):
                continue

            # Exclude quarterly and year-to-date values.
            if not 330 <= (end - start).days <= 390:
                continue

            fact = {
                "value_usd": value,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }

            if end == report_date:
                current.append(fact)

            elif 330 <= (report_date - end).days <= 390:
                previous.append(fact)

        # Remove exact duplicates without discarding conflicts.
        current = {
            (f["start"], f["end"], f["value_usd"]): f
            for f in current
        }

        previous = {
            (f["start"], f["end"], f["value_usd"]): f
            for f in previous
        }

        if len(current) > 1 or len(previous) > 1:
            raise ValueError(
                f"Conflicting annual financial values for {tag}"
            )

        if current and previous:
            return (
                tag,
                next(iter(current.values())),
                next(iter(previous.values())),
            )

    return None


def get_financial_comparison(
    ticker: str,
    cik: str | int,
    fiscal_year: int,
) -> dict:
    """Compare annual SEC financial facts with source tracking."""

    # 1. Validate inputs.
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker cannot be empty")

    if type(fiscal_year) is not int or not 2009 <= fiscal_year <= 2100:
        raise ValueError("Invalid fiscal year")

    ticker = ticker.strip().upper()
    cik = normalize_cik(cik)

    # 2. Reuse existing annual filing resolver.
    accession = resolve_annual_filing(
        f"{ticker} fiscal year {fiscal_year}",
        ticker,
    )

    if not accession:
        raise ValueError(
            f"No indexed annual filing found for {ticker} FY{fiscal_year}"
        )

    # 3. Retrieve filing metadata from PostgreSQL.
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                f.report_date,
                (
                    SELECT c.source_url
                    FROM chunks c
                    WHERE c.accession_number = f.accession_number
                    LIMIT 1
                )
            FROM filings f
            WHERE f.accession_number = %s
              AND f.ticker = %s
              AND f.filing_type = '10-K'
            """,
            (accession, ticker),
        ).fetchone()

    if row is None or row[0] is None:
        raise ValueError(
            "Annual filing or reporting date missing from PostgreSQL"
        )

    report_date, filing_url = row

    # 4. Reuse the existing SEC API client.
    client = SECClient()

    try:
        raw = client.get_companyfacts(cik)
    finally:
        client.close()

    companyfacts = json.loads(raw)

    if (
        not isinstance(companyfacts, dict)
        or normalize_cik(companyfacts.get("cik")) != cik
    ):
        raise ValueError("SEC Company Facts CIK mismatch")

    # 5. Preserve a SHA-256 verified source snapshot.
    source_sha256 = hashlib.sha256(raw).hexdigest()

    snapshot = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "sec"
        / cik
        / "companyfacts"
        / f"{source_sha256}.json"
    )

    if not snapshot.exists():
        write_atomic(snapshot, raw)

    # 6. Extract financial metrics and calculate changes.
    metrics = {}

    for name, tags in METRIC_TAGS.items():
        pair = _annual_pair(
            companyfacts,
            accession,
            report_date,
            tags,
        )

        if pair is None:
            if name == "revenue":
                raise ValueError(
                    "Comparable annual revenue facts not found"
                )

            # Other accounting metrics may be unavailable.
            continue

        tag, current, previous = pair

        current_value = current["value_usd"]
        previous_value = previous["value_usd"]

        change = current_value - previous_value

        # A non-positive baseline does not have a meaningful
        # conventional percentage-growth interpretation.
        if previous_value > 0:
            growth = float(
                (
                    Decimal(change)
                    * 100
                    / Decimal(previous_value)
                ).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )
            )
        else:
            growth = None

        metrics[name] = {
            "concept": f"us-gaap:{tag}",
            "current": current,
            "previous": previous,
            "change_usd": change,
            "change_percent": growth,
        }

    # 7. Return calculations with full source metadata.
    return {
        "ticker": ticker,
        "cik": cik,
        "fiscal_year": fiscal_year,
        "accession_number": accession,
        "report_date": report_date.isoformat(),
        "metrics": metrics,
        "source": {
            "filing_url": filing_url,
            "companyfacts_url": (
                "https://data.sec.gov/api/xbrl/companyfacts/"
                f"CIK{cik}.json"
            ),
            "sha256": source_sha256,
            "snapshot_path": str(
                snapshot.relative_to(PROJECT_ROOT)
            ),
        },
    }


def main():
    """Run NVIDIA annual financial analysis locally."""

    parser = argparse.ArgumentParser(
        description="Compare NVIDIA annual SEC financial facts"
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Fiscal year to compare with the preceding fiscal year",
    )

    args = parser.parse_args()

    result = get_financial_comparison(
        ticker="NVDA",
        cik="0001045810",
        fiscal_year=args.year,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

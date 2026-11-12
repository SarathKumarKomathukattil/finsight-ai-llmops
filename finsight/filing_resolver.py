
"""Resolve an annual SEC filing from a financial question."""

import re
from datetime import date

from finsight.database import get_connection


def resolve_annual_filing(question: str, ticker: str = "NVDA") -> str | None:
    """Find an annual filing for a requested fiscal year."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty")

    # Identify expressions such as FY2026 or fiscal year 2026.
    years = set(
        re.findall(
            r"\b(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\b",
            question,
            flags=re.IGNORECASE,
        )
    )

    if not years:
        return None

    if len(years) != 1:
        raise ValueError("Multiple fiscal years need separate filing selection")

    year = int(years.pop())

    # Initial implementation: annual filings whose reporting-period
    # ending year matches the fiscal-year label.
    with get_connection() as conn:
        matches = conn.execute(
            """
            SELECT accession_number
            FROM filings
            WHERE ticker = %s
              AND filing_type = '10-K'
              AND report_date >= %s
              AND report_date < %s
            ORDER BY filing_date DESC
            LIMIT 2
            """,
            (
                ticker,
                date(year, 1, 1),
                date(year + 1, 1, 1),
            ),
        ).fetchall()

    if not matches:
        raise ValueError(f"No indexed annual filing found for {ticker} FY{year}")

    if len(matches) > 1:
        raise ValueError(f"Multiple annual filings found for {ticker} FY{year}")

    return matches[0][0]


from mcp.server import MCPServer

from finsight.financial_facts import get_financial_comparison
from finsight.gdelt_client import fetch_news


mcp = MCPServer("FinSight AI")


@mcp.tool()
def get_annual_financials(
    ticker: str,
    cik: str,
    fiscal_year: int,
) -> dict:
    """Get verified annual financial facts and period comparisons from SEC data."""

    if not ticker.strip() or not cik.strip():
        raise ValueError("Ticker and CIK are required")

    return get_financial_comparison(
        ticker=ticker.strip().upper(),
        cik=cik.strip(),
        fiscal_year=fiscal_year,
    )


@mcp.tool()
def get_recent_news(
    ticker: str = "NVDA",
    days: int = 7,
    limit: int = 5,
) -> dict:
    """Retrieve recent company news using the existing news connector."""

    if not ticker.strip():
        raise ValueError("Ticker is required")

    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")

    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")

    return fetch_news(
        ticker=ticker.strip().upper(),
        days=days,
        limit=limit,
    )


if __name__ == "__main__":
    mcp.run()

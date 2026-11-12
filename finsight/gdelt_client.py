
"""Fetch relevant financial news with a fallback provider."""

import argparse
import json
import logging
import re
import time

from datetime import datetime, timezone
from urllib.parse import urlparse, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx


logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_RSS_URL = "https://news.google.com/rss/search"

COMPANIES = {
    "NVDA": '"NVIDIA"',
}

# Words indicating potentially relevant business or financial news.
FINANCIAL_TERMS = (
    "stock",
    "share",
    "earnings",
    "revenue",
    "profit",
    "sales",
    "forecast",
    "guidance",
    "growth",
    "market",
    "investment",
    "investor",
    "valuation",
    "financial",
    "quarter",
    "data center",
    "datacenter",
    "chip",
    "gpu",
    "artificial intelligence",
    "export",
    "regulation",
    "regulatory",
    "antitrust",
    "supply",
    "demand",
    "acquisition",
    "merger",
    "deal",
    "contract",
    "partnership",
    "product launch",
    "leadership",
    "succession",
    "business risk",
)

# Avoid repeatedly contacting GDELT after rate limiting.
_gdelt_cooldown_until = 0.0


class NewsFetchError(RuntimeError):
    """Raised when both news providers are unavailable."""


def _valid_link(url):
    """Accept only valid HTTPS article links."""

    if not isinstance(url, str):
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
    )


def _select_articles(articles, ticker, limit):
    """Deduplicate and prioritize financially relevant headlines."""

    company = COMPANIES[ticker].strip('"').lower()

    selected = []
    seen_urls = set()
    seen_titles = set()

    for article in articles:
        title = article["title"]
        url = article["url"]

        title_lower = title.lower()

        # The headline must mention the tracked company.
        if company not in title_lower and ticker.lower() not in title_lower:
            continue

        # Require at least one business/financial topic.
        relevant = any(
            re.search(
                rf"\b{re.escape(term)}\b",
                title_lower,
            )
            for term in FINANCIAL_TERMS
        )

        if not relevant:
            continue

        # Ignore tracking query parameters when comparing URLs.
        parts = urlsplit(url)

        normalized_url = urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            "",
            "",
        ))

        # Normalize punctuation and spacing for title comparison.
        normalized_title = re.sub(
            r"[\W_]+",
            "",
            title_lower,
        )

        if (
            normalized_url in seen_urls
            or normalized_title in seen_titles
        ):
            continue

        seen_urls.add(normalized_url)
        seen_titles.add(normalized_title)

        selected.append(article)

        if len(selected) >= limit:
            break

    return selected


def _fetch_gdelt(client, query, days, limit):
    """Retrieve relevant article metadata from GDELT."""

    response = client.get(
        GDELT_URL,
        params={
            "query": (
                f"{query} "
                "(stock OR earnings OR revenue OR chips "
                "OR export OR market OR investment "
                "OR regulation OR leadership OR deal) "
                "sourcelang:english"
            ),
            "mode": "artlist",
            "format": "json",
            "sort": "hybridrel",
            "timespan": f"{days}d",
            "maxrecords": min(250, max(20, limit * 4)),
        },
    )

    response.raise_for_status()

    data = response.json()

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("articles"), list)
    ):
        raise ValueError("Invalid GDELT response")

    articles = []

    for item in data["articles"]:
        if not isinstance(item, dict):
            continue

        url = item.get("url")
        title = item.get("title")

        if (
            not _valid_link(url)
            or not isinstance(title, str)
            or not title.strip()
        ):
            continue

        articles.append({
            "title": title.strip(),
            "url": url,
            "domain": item.get("domain"),
            "seen_at": item.get("seendate"),
            "published_at": None,
            "language": item.get("language"),
            "source_country": item.get("sourcecountry"),
        })

    return articles, str(response.url)


def _fetch_google_news(client, query, days, limit):
    """Retrieve article metadata from Google News RSS."""

    response = client.get(
        GOOGLE_RSS_URL,
        params={
            "q": f"{query} when:{days}d",
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        },
    )

    response.raise_for_status()

    if len(response.content) > 3_000_000:
        raise ValueError("News feed is unexpectedly large")

    root = ElementTree.fromstring(response.content)

    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("Invalid Google News RSS feed")

    articles = []

    for item in root.findall("./channel/item"):
        url = item.findtext("link")
        title = item.findtext("title")

        if (
            not _valid_link(url)
            or not title
            or not title.strip()
        ):
            continue

        publisher = item.find("source")

        publisher_url = (
            publisher.get("url", "")
            if publisher is not None
            else ""
        )

        articles.append({
            "title": title.strip(),
            "url": url,
            "domain": urlparse(publisher_url).hostname,
            "seen_at": None,
            "published_at": item.findtext("pubDate"),
            "language": "English",
            "source_country": None,
        })

    return articles, str(response.url)


def fetch_news(
    ticker: str = "NVDA",
    days: int = 7,
    limit: int = 10,
) -> dict:
    """Fetch financial news without storing or embedding articles."""

    global _gdelt_cooldown_until

    if not isinstance(ticker, str):
        raise ValueError("Invalid ticker")

    ticker = ticker.strip().upper()

    if ticker not in COMPANIES:
        raise ValueError(f"Unsupported ticker: {ticker}")

    if type(days) is not int or not 1 <= days <= 90:
        raise ValueError("Days must be between 1 and 90")

    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Limit must be between 1 and 50")

    query = COMPANIES[ticker]

    with httpx.Client(
        timeout=25.0,
        follow_redirects=False,
    ) as client:

        # 1. Try GDELT first.
        if time.monotonic() >= _gdelt_cooldown_until:

            try:
                candidates, url = _fetch_gdelt(
                    client,
                    query,
                    days,
                    limit,
                )

                articles = _select_articles(
                    candidates,
                    ticker,
                    limit,
                )

                if articles:
                    return {
                        "ticker": ticker,
                        "query": query,
                        "retrieved_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "article_count": len(articles),
                        "articles": articles,
                        "source": {
                            "provider": "GDELT DOC 2.0",
                            "api_url": url,
                        },
                    }

                logger.warning(
                    "No relevant GDELT articles. Trying RSS."
                )

            except httpx.HTTPStatusError as exc:

                if exc.response.status_code == 429:

                    retry_after = exc.response.headers.get(
                        "Retry-After", ""
                    )

                    try:
                        cooldown = min(
                            3600,
                            max(300, int(retry_after)),
                        )
                    except ValueError:
                        cooldown = 900

                    _gdelt_cooldown_until = (
                        time.monotonic() + cooldown
                    )

                    logger.warning(
                        "GDELT rate-limited. Using Google News RSS."
                    )

                else:
                    logger.warning(
                        "GDELT HTTP %s. Using RSS fallback.",
                        exc.response.status_code,
                    )

            except (httpx.RequestError, ValueError) as exc:

                logger.warning(
                    "GDELT unavailable (%s). Using RSS fallback.",
                    type(exc).__name__,
                )

        # 2. Fall back to Google News RSS.
        try:
            candidates, url = _fetch_google_news(
                client,
                query,
                days,
                limit,
            )

            articles = _select_articles(
                candidates,
                ticker,
                limit,
            )

        except (
            httpx.HTTPError,
            ValueError,
            ElementTree.ParseError,
        ) as exc:
            raise NewsFetchError(
                "Both GDELT and Google News RSS are unavailable"
            ) from exc

    # 3. Preserve provider metadata and article provenance.
    return {
        "ticker": ticker,
        "query": query,
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "article_count": len(articles),
        "articles": articles,
        "source": {
            "provider": "Google News RSS",
            "api_url": url,
        },
    }


def main():
    """Run live financial news retrieval."""

    parser = argparse.ArgumentParser(
        description="Retrieve current financial news"
    )

    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    try:
        result = fetch_news(
            ticker=args.ticker,
            days=args.days,
            limit=args.limit,
        )

    except (ValueError, NewsFetchError) as exc:
        parser.exit(
            1,
            f"News retrieval failed: {exc}\n",
        )

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

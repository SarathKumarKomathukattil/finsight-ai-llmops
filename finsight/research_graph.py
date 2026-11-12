
"""FinSight financial research workflow using LangGraph and MCP."""

import asyncio
import logging
import re
from typing import TypedDict
import json
from mcp.types import TextContent

import torch
from mcp import Client
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer, CrossEncoder

from finsight.answer_generator import generate_answer
from finsight.filing_resolver import resolve_annual_filing
from finsight.index_filings import MODEL_NAME
from finsight.mcp_server import mcp
from finsight.retrieval import retrieve_chunks, RERANKER_NAME


logger = logging.getLogger(__name__)


# --------------------------------------------------
# MCP client
# --------------------------------------------------


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Call a FinSight MCP tool and return dictionary data."""

    async with Client(mcp) as client:
        result = await client.call_tool(tool_name, arguments)

    if result.is_error:
        raise RuntimeError(
            f"MCP tool '{tool_name}' failed: {result.content}"
        )

    # Prefer structured MCP output.
    data = result.structured_content

    if isinstance(data, dict):
        # Some MCP outputs wrap a value under "result".
        if (
            set(data) == {"result"}
            and isinstance(data["result"], dict)
        ):
            return data["result"]

        return data

    # Fallback for MCP tools returning JSON as text.
    for block in result.content:
        if isinstance(block, TextContent):
            try:
                data = json.loads(block.text)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                return data

    raise RuntimeError(
        f"MCP tool '{tool_name}' returned no valid dictionary"
    )



# --------------------------------------------------
# LangGraph state
# --------------------------------------------------

class ResearchState(TypedDict, total=False):
    """Shared state for the financial research workflow."""

    question: str
    ticker: str
    wants_news: bool
    news_only: bool
    accession_number: str | None
    evidence: list[dict]
    financial_facts: dict | None
    news: dict | None
    news_error: str | None
    answer: str
    sources: list[dict]
    status: str


# --------------------------------------------------
# Build research graph
# --------------------------------------------------

def build_research_graph(model, reranker):
    """Build the financial research workflow."""

    # ----------------------------------------------
    # Node 1: Route question
    # ----------------------------------------------

    def route_question(state: ResearchState) -> dict:
        """Determine whether the question needs SEC data or news."""

        question = state["question"].lower()

        if state["ticker"].upper() != "NVDA":
            raise ValueError("Only NVDA is currently supported")

        news_terms = (
            "news",
            "headline",
            "breaking",
            "recent developments",
        )

        financial_terms = (
            "revenue",
            "growth",
            "profit",
            "income",
            "earnings",
            "fiscal",
            "filing",
            "quarter",
            "10-k",
            "10-q",
            "risk",
            "export",
            "supply chain",
            "financial",
            "compare",
        )

        wants_news = any(
            term in question for term in news_terms
        )

        wants_financial_data = (
            any(term in question for term in financial_terms)
            or bool(re.search(r"\bfy\s*20\d{2}\b", question))
        )

        if (
            ("latest" in question or "today" in question)
            and not wants_financial_data
        ):
            wants_news = True

        news_only = wants_news and not wants_financial_data

        return {
            "wants_news": wants_news,
            "news_only": news_only,
        }

    # ----------------------------------------------
    # Node 2: Resolve SEC filing
    # ----------------------------------------------

    def resolve_filing(state: ResearchState) -> dict:
        """Identify the correct SEC filing."""

        accession = resolve_annual_filing(
            state["question"],
            state["ticker"],
        )

        logger.info(
            "Selected SEC accession: %s",
            accession,
        )

        return {
            "accession_number": accession,
        }

    # ----------------------------------------------
    # Node 3: Retrieve SEC evidence
    # ----------------------------------------------

    def retrieve_evidence(state: ResearchState) -> dict:
        """Retrieve and rerank relevant SEC passages."""

        results = retrieve_chunks(
            state["question"],
            model,
            reranker,
            ticker=state["ticker"],
            accession_number=state.get("accession_number"),
        )

        evidence = [
            {
                "id": f"S{i}",
                "chunk_id": row[0],
                "text": row[1],
                "heading": row[2],
                "source_url": row[3],
                "accession_number": row[4],
            }
            for i, (row, _) in enumerate(
                results[:5],
                start=1,
            )
        ]

        logger.info(
            "Retrieved %d SEC passages",
            len(evidence),
        )

        return {
            "evidence": evidence,
        }

    # ----------------------------------------------
    # Node 4: Retrieve financial facts through MCP
    # ----------------------------------------------

    async def load_financial_facts(
        state: ResearchState,
    ) -> dict:
        """Retrieve verified SEC financial facts using MCP."""

        question = state["question"]

        match = re.search(
            r"\b(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\b",
            question,
            flags=re.IGNORECASE,
        )

        financial_terms = (
            "revenue",
            "growth",
            "profit",
            "income",
            "earnings",
            "sales",
        )

        if (
            not match
            or not any(
                term in question.lower()
                for term in financial_terms
            )
        ):
            return {
                "financial_facts": None,
            }

        facts = await call_mcp_tool(
            "get_annual_financials",
            {
                "ticker": state["ticker"],
                "cik": "0001045810",
                "fiscal_year": int(match.group(1)),
            },
        )

        if (
            facts["accession_number"]
            != state.get("accession_number")
        ):
            raise ValueError(
                "Financial facts filing mismatch"
            )

        logger.info(
            "Loaded verified SEC XBRL financial facts through MCP"
        )

        return {
            "financial_facts": facts,
        }

    # ----------------------------------------------
    # Node 5: Retrieve news through MCP
    # ----------------------------------------------

    async def retrieve_live_news(
        state: ResearchState,
    ) -> dict:
        """Retrieve live company news using MCP."""

        if not state.get("wants_news", False):
            return {
                "news": None,
                "news_error": None,
            }

        try:
            news = await call_mcp_tool(
                "get_recent_news",
                {
                    "ticker": state["ticker"],
                    "days": 7,
                    "limit": 5,
                },
            )

            logger.info(
                "Retrieved %d articles through MCP",
                news["article_count"],
            )

            return {
                "news": news,
                "news_error": None,
            }

        except (RuntimeError, KeyError, TypeError) as exc:
            logger.warning(
                "Live news unavailable: %s",
                exc,
            )

            # News failure must not stop SEC research.
            return {
                "news": None,
                "news_error": (
                    "Live news is temporarily unavailable."
                ),
            }

    # ----------------------------------------------
    # Node 6: Generate financial answer
    # ----------------------------------------------

    def generate_research_answer(
        state: ResearchState,
    ) -> dict:
        """Generate an evidence-backed financial response."""

        evidence = state.get("evidence", [])
        facts = state.get("financial_facts")
        news = state.get("news")

        news_count = (
            news["article_count"]
            if news
            else 0
        )

        # Handle news-only questions.
        if state.get("news_only", False):

            if news_count:
                return {
                    "answer": (
                        "Recent NVIDIA news headlines are "
                        "available in the news section. "
                        "Article contents have not been "
                        "independently verified."
                    ),
                    "sources": [],
                    "status": "answered",
                }

            return {
                "answer": "No recent news could be retrieved.",
                "sources": [],
                "status": "insufficient_evidence",
            }

        # Handle missing SEC evidence.
        if not evidence and not facts:
            return {
                "answer": (
                    "No matching SEC evidence was found. "
                    "Recent news, if available, is listed separately."
                ),
                "sources": [],
                "status": (
                    "answered"
                    if news_count
                    else "insufficient_evidence"
                ),
            }

        answer = ""
        sources = []

        # Generate narrative answer from SEC passages.
        if evidence:

            answer, sources = generate_answer(
                state["question"],
                evidence,
            )

            if not sources:
                answer = ""

        # Add verified XBRL financial calculations.
        if facts:

            revenue = facts["metrics"]["revenue"]
            year = facts["fiscal_year"]

            current = (
                revenue["current"]["value_usd"] / 1e9
            )

            previous = (
                revenue["previous"]["value_usd"] / 1e9
            )

            change = (
                revenue["change_usd"] / 1e9
            )

            growth = revenue["change_percent"]

            growth_text = (
                f"{growth:+.2f}%"
                if growth is not None
                else "not meaningful"
            )

            verified_summary = (
                f"Verified SEC XBRL figures: "
                f"FY{year} revenue was "
                f"${current:,.3f} billion, "
                f"compared with "
                f"${previous:,.3f} billion "
                f"in FY{year - 1}. "
                f"Revenue changed by "
                f"${change:,.3f} billion "
                f"({growth_text} year over year). [F1]"
            )

            sources.insert(
                0,
                {
                    "id": "F1",
                    "chunk_id": (
                        f"xbrl:"
                        f"{facts['accession_number']}:revenue"
                    ),
                    "heading": (
                        "SEC XBRL annual revenue facts"
                    ),
                    "source_url": (
                        facts["source"]["companyfacts_url"]
                    ),
                    "accession_number": (
                        facts["accession_number"]
                    ),
                },
            )

            answer = (
                verified_summary + "\n\n" + answer
                if answer
                else verified_summary
            )

        return {
            "answer": answer,
            "sources": sources,
            "status": (
                "answered"
                if sources or news_count
                else "insufficient_evidence"
            ),
        }

    # ----------------------------------------------
    # Construct LangGraph workflow
    # ----------------------------------------------

    workflow = StateGraph(ResearchState)

    workflow.add_node(
        "route_question",
        route_question,
    )

    workflow.add_node(
        "resolve_filing",
        resolve_filing,
    )

    workflow.add_node(
        "retrieve_evidence",
        retrieve_evidence,
    )

    workflow.add_node(
        "load_financial_facts",
        load_financial_facts,
    )

    workflow.add_node(
        "retrieve_live_news",
        retrieve_live_news,
    )

    workflow.add_node(
        "generate_answer",
        generate_research_answer,
    )

    workflow.add_edge(
        START,
        "route_question",
    )

    # Skip SEC retrieval for news-only questions.
    workflow.add_conditional_edges(
        "route_question",
        lambda state: (
            "news"
            if state["news_only"]
            else "sec"
        ),
        {
            "news": "retrieve_live_news",
            "sec": "resolve_filing",
        },
    )

    workflow.add_edge(
        "resolve_filing",
        "retrieve_evidence",
    )

    workflow.add_edge(
        "retrieve_evidence",
        "load_financial_facts",
    )

    workflow.add_edge(
        "load_financial_facts",
        "retrieve_live_news",
    )

    workflow.add_edge(
        "retrieve_live_news",
        "generate_answer",
    )

    workflow.add_edge(
        "generate_answer",
        END,
    )

    return workflow.compile()


# --------------------------------------------------
# Command-line execution
# --------------------------------------------------

async def main():
    """Run the financial research workflow."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    question = input(
        "\nAsk a financial question: "
    ).strip()

    if not question:
        raise SystemExit(
            "Question cannot be empty"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nModel device: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    reranker = CrossEncoder(
        RERANKER_NAME,
        device=device,
    )

    graph = build_research_graph(
        model,
        reranker,
    )

    try:
        result = await graph.ainvoke(
            {
                "question": question,
                "ticker": "NVDA",
            }
        )

    except Exception:
        logger.exception(
            "Financial research workflow failed"
        )
        raise SystemExit(1)

    print("\nSTATUS:", result["status"])

    print("\nFINANCIAL ANSWER:\n")
    print(result["answer"])

    print("\nSEC SOURCES:")

    for source in result.get("sources", []):
        print(
            f"{source['id']}: "
            f"{source['source_url']}"
        )

    # Display news separately from SEC evidence.
    news = result.get("news")

    if news:
        print("\nRECENT FINANCIAL NEWS")
        print(
            "Provider:",
            news["source"]["provider"],
        )
        print(
            "Retrieved:",
            news["retrieved_at"],
        )

        for article in news["articles"]:

            print(
                f"\nTitle: {article['title']}"
            )

            print(
                "Date:",
                article.get("published_at")
                or article.get("seen_at"),
            )

            print(
                f"URL: {article['url']}"
            )

    if result.get("news_error"):
        print(
            "\nNEWS WARNING:",
            result["news_error"],
        )


if __name__ == "__main__":
    asyncio.run(main())

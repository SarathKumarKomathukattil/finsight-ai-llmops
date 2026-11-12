
"""FastAPI application for FinSight financial research."""

import logging
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal
import mlflow
import httpx
import psycopg
import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sentence_transformers import SentenceTransformer, CrossEncoder

from finsight.answer_generator import AnswerGenerationError
from finsight.index_filings import MODEL_NAME
from finsight.retrieval import RERANKER_NAME
from finsight.research_graph import build_research_graph
from finsight.sec_client import RetryDeferredError


logger = logging.getLogger(__name__)

# Restrict concurrent local GPU inference.
inference_lock = Lock()


class ResearchRequest(BaseModel):
    """Validate incoming research questions."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=10, max_length=500)
    ticker: str = Field(default="NVDA", min_length=1, max_length=10)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()

        if ticker != "NVDA":
            raise ValueError("Only NVDA is currently supported")

        return ticker


class SourceReference(BaseModel):
    """Citation for an SEC filing or XBRL source."""

    id: str
    chunk_id: str
    heading: str
    source_url: str
    accession_number: str


class NewsArticle(BaseModel):
    """Live news article metadata."""

    title: str
    url: str
    domain: str | None = None
    seen_at: str | None = None
    published_at: str | None = None
    language: str | None = None
    source_country: str | None = None


class NewsProvider(BaseModel):
    """News provider information."""

    provider: str
    api_url: str


class NewsResponse(BaseModel):
    """Structured live news response."""

    ticker: str
    query: str
    retrieved_at: str
    article_count: int
    articles: list[NewsArticle]
    source: NewsProvider


class ResearchResponse(BaseModel):
    """Complete SEC and news research response."""

    ticker: str
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    sources: list[SourceReference]
    news: NewsResponse | None = None
    news_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding and reranking models once at startup."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Starting FinSight AI on %s", device)

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    reranker = CrossEncoder(
        RERANKER_NAME,
        device=device,
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("FinSight-Financial-Research")
    mlflow.langchain.autolog()

    app.state.research_graph = build_research_graph(
        model,
        reranker,
    )

    logger.info("FinSight research workflow initialized")

    try:
        yield
    finally:
        app.state.research_graph = None
        logger.info("FinSight research API stopped")


app = FastAPI(
    title="FinSight AI",
    description=(
        "SEC-grounded financial research with live news, "
        "verified XBRL calculations, and LangGraph."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Return API health status."""

    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Execute the FinSight financial research workflow."""

    # Prevent overlapping local inference requests.
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Research engine is busy. Retry shortly.",
        )

    try:
        graph = app.state.research_graph

        if graph is None:
            raise HTTPException(
                status_code=503,
                detail="Research engine is not ready.",
            )

        result = await graph.ainvoke({
            "question": request.question,
            "ticker": request.ticker,
        })

        return ResearchResponse(
            ticker=request.ticker,
            status=result["status"],
            answer=result["answer"],
            sources=[
                SourceReference.model_validate(source)
                for source in result.get("sources", [])
            ],
            news=(
                NewsResponse.model_validate(result["news"])
                if result.get("news") is not None
                else None
            ),
            news_error=result.get("news_error"),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except AnswerGenerationError as exc:
        logger.exception("Financial answer generation failed")

        raise HTTPException(
            status_code=502,
            detail="Local LLM answer generation failed.",
        ) from exc

    except (
        psycopg.Error,
        httpx.HTTPError,
        RetryDeferredError,
        ConnectionError,
    ) as exc:
        logger.exception("External service or database unavailable")

        raise HTTPException(
            status_code=503,
            detail="A required research service is unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected financial research failure")

        raise HTTPException(
            status_code=500,
            detail="Internal research error.",
        ) from exc

    finally:
        inference_lock.release()

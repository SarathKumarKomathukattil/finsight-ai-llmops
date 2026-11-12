# FinSight AI — Financial Research & LLMOps

An evidence-backed financial research application built with **Python, FastAPI, LangGraph, MCP, SEC EDGAR/XBRL, PostgreSQL/pgvector, Sentence Transformers, Ollama and MLflow**.

## What it does

- Ingests and parses real SEC filings with source provenance and checksums.
- Retrieves and reranks relevant filing passages for financial RAG.
- Calculates annual revenue comparisons from structured SEC XBRL facts (not LLM arithmetic).
- Orchestrates SEC research and optional news retrieval through LangGraph and MCP tools.
- Uses GDELT for recent news, with Google News RSS fallback.
- Returns structured answers and original SEC citations through `POST /research`.
- Captures LangGraph execution traces in MLflow.

**Current scope:** NVIDIA (NVDA), local inference and local database. Kafka, Airflow, AKS, KEDA, vLLM and AIOps are planned infrastructure extensions, not implemented production services.

## Demonstrated result

For “What factors contributed to NVIDIA's revenue growth in fiscal 2026?” the application returned FY2026 revenue of **$215.938B**, FY2025 revenue of **$130.497B**, and **65.47%** year-over-year growth with sources:

- [SEC XBRL company facts](https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json)
- [NVIDIA FY2026 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)

Local validation on November 13, 2026: **24 pytest tests passed**. A successful MLflow trace recorded **9.93 seconds** end-to-end in one run. These are local results, not cloud CI benchmarks.

## Run locally

Requires Python 3.12, configured PostgreSQL/pgvector, a running Ollama instance with the project's configured LLM, and the application environment variables. Install dependencies from `requirements.txt` after the source and dependency manifest are published.

```powershell
python -m mlflow server --host 127.0.0.1 --port 5000
```

In a second terminal:

```powershell
python -m uvicorn finsight.api:app --host 127.0.0.1 --port 8000
```

Open [FastAPI Swagger UI](http://127.0.0.1:8000/docs), then call `POST /research`:

```json
{
  "question": "What factors contributed to NVIDIA's revenue growth in fiscal 2026?",
  "ticker": "NVDA"
}
```

Run the existing tests with:

```powershell
python -m pytest tests/ -q
```

### CI

The [GitHub Actions workflow](.github/workflows/ci.yml) runs syntax checks and pytest on Python 3.12 once the source code and `requirements.txt` are pushed. Remote CI is not considered passing until GitHub displays a green run.

## Engineering notes

Financial figures originate from SEC facts, not model-generated calculations. SEC narrative claims are cited to filing passages. Live news is metadata, not independently verified article content. The repository is a research engineering demonstration, not a trading or personalized investment-advice service.

## Roadmap

Kafka event ingestion, Airflow reconciliation DAGs, MLflow evaluation gates, Docker and Kubernetes, Azure AKS/Terraform, vLLM serving, KEDA autoscaling, Prometheus/Grafana and operational AIOps.

See [LICENSE](LICENSE).

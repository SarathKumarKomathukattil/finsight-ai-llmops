"""Benchmark one SEC-grounded research response through the existing FastAPI API."""

import time
from urllib.parse import urlparse

import httpx
import mlflow

API_URL = "http://127.0.0.1:8000/research"
MLFLOW_URL = "http://127.0.0.1:5000"
QUESTION = "What factors contributed to NVIDIA's revenue growth in fiscal 2026?"
EXPECTED_VALUES = ("$215.938 billion", "$130.497 billion", "$85.441 billion", "+65.47%")
EXPECTED_ACCESSION = "0001045810-26-000021"


def main():
    """Run a deterministic, single-question financial regression benchmark."""
    started = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            API_URL,
            json={"question": QUESTION, "ticker": "NVDA"},
        )
        response.raise_for_status()

    latency = time.perf_counter() - started
    result = response.json()
    answer = result.get("answer", "")
    sources = result.get("sources", [])

    value_checks = {value: value in answer for value in EXPECTED_VALUES}
    source_ids = {source.get("id") for source in sources}

    citation_valid = bool(sources) and all(
        source.get("accession_number") == EXPECTED_ACCESSION
        and urlparse(source.get("source_url", "")).hostname
        in {"www.sec.gov", "data.sec.gov"}
        for source in sources
    ) and all(
        f"[{source_id}]" in answer and source_id in source_ids
        for source_id in ("F1", "S1")
    )

    metrics = {
        "status_correct": int(result.get("status") == "answered"),
        "expected_values_present": sum(value_checks.values()) / len(value_checks),
        "citation_metadata_valid": int(citation_valid),
        "latency_seconds": latency,
    }

    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment("FinSight-Research-Benchmark")

    with mlflow.start_run(run_name="nvda-fy2026-baseline"):
        mlflow.log_param("ticker", "NVDA")
        mlflow.log_param("fiscal_year", 2026)
        mlflow.log_param("question_count", 1)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "question": QUESTION,
                "expected_values": list(EXPECTED_VALUES),
                "value_checks": value_checks,
                "metrics": metrics,
                "response": result,
            },
            "research_evaluation.json",
        )

    print("\n=== FinSight Research Benchmark ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\nResults saved to MLflow.")


if __name__ == "__main__":
    main()

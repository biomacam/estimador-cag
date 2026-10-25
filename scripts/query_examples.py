#!/usr/bin/env python3
"""Ingest the example corpus and run representative queries through POST /search.

Usage from the project root::

    docker compose up -d
    docker compose run --rm estimator python scripts/query_examples.py

Compose sets ESTIMATOR_API_BASE_URL=http://estimator:8000 for the one-off
container so it calls the running API instead of its own localhost.

Or run from the host with the API available at localhost:8000::

    uv run python scripts/query_examples.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

CORPUS_PATH = ROOT / "data" / "budgets_sample.json"
TOP_K = 5
CONTENT_PREVIEW_CHARS = 120
QUERIES = (
    (
        "Componente directo conocido (sanity check)",
        "REST API development with JWT authentication for financial sector",
    ),
    (
        "Reformulacion semantica (mismo concepto, otro vocabulario)",
        "secure backend service with token-based access control for banking applications",
    ),
    (
        "Dominio distinto (no deberia estar en el corpus)",
        "mobile application for restaurant reservations",
    ),
    (
        "Consulta ambigua (sin match dominante)",
        "integration with external system",
    ),
    (
        "Consulta muy especifica (vocabulario tecnico preciso)",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
)


def resolve_base_url(client: httpx.Client) -> str:
    """Probe the configured URL and common Compose/host addresses."""
    explicit = os.environ.get("ESTIMATOR_BASE_URL")
    if explicit:
        candidates = (explicit,)
    else:
        configured = get_settings().ESTIMATOR_API_BASE_URL
        candidates = tuple(dict.fromkeys((configured, "http://localhost:8000", "http://estimator:8000")))

    for base_url in candidates:
        try:
            if client.get(f"{base_url}/health").status_code == 200:
                return base_url
        except httpx.TransportError:
            continue

    print(
        "ERROR: no estimator API reachable. Start the stack (docker compose up -d) "
        "or set ESTIMATOR_BASE_URL.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def ingest_corpus(client: httpx.Client, base_url: str) -> None:
    """Ingest one source document per example budget; 409 means already present."""
    budgets = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    created = 0
    skipped = 0
    for budget in budgets:
        response = client.post(
            f"{base_url}/embeddings/ingest",
            json={
                "source_path": f"data/budgets_sample.json::{budget['budget_id']}",
                "document_type": "historical_budget",
                "content": budget,
            },
        )
        if response.status_code == 200:
            created += 1
        elif response.status_code == 409:
            skipped += 1
        else:
            print(
                f"ERROR ingesting {budget['budget_id']}: "
                f"{response.status_code} {response.text[:200]}",
                file=sys.stderr,
            )
            raise SystemExit(1)

    print(f"Corpus: {len(budgets)} budgets - {created} ingested, {skipped} already present.")


def run_queries(client: httpx.Client, base_url: str) -> None:
    for index, (label, query) in enumerate(QUERIES, start=1):
        response = client.post(f"{base_url}/search", json={"query": query, "k": TOP_K})
        response.raise_for_status()
        body = response.json()

        print()
        print(f"[{index}/{len(QUERIES)}] {label}")
        print(f'    query: "{query}"')
        print(f"    search_time_ms: {body['search_time_ms']}")
        print(f"    {'chunk_id':>8}  {'distance':>8}  {'chunk_type':<18}  content")
        for result in body["results"]:
            content = " ".join(result["content"].split())[:CONTENT_PREVIEW_CHARS]
            print(
                f"    {result['chunk_id']:>8}  {result['distance']:>8.4f}  "
                f"{result['chunk_type']:<18}  {content}"
            )


def main() -> int:
    with httpx.Client(timeout=120.0) as client:
        base_url = resolve_base_url(client)
        print(f"Estimator API: {base_url}")
        ingest_corpus(client, base_url)
        run_queries(client, base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

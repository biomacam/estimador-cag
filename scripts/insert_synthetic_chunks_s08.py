#!/usr/bin/env python3
"""Insert synthetic chunks with REAL embeddings to simulate corpus growth.

Session 08 — two uses:

* Maintenance demo (default ``count=100``): simulate post-launch growth before
  ANALYZE / VACUUM / REINDEX CONCURRENTLY.
* Pre-flight (e.g. ``count=30000``): fatten the corpus so the sequential-scan
  baseline is measurable and the HNSW before/after gap is visible. With the
  real sample corpus a sequential scan is sub-millisecond.

Embeddings are real ``text-embedding-3-small`` vectors: random vectors do not
share the corpus geometry, so recall sweeps against them would be meaningless.

All rows hang from one synthetic document (``document_type = 'synthetic_test'``)
and are tagged ``chunk_type = 'synthetic'``. Clean-up::

    DELETE FROM documents WHERE document_type = 'synthetic_test';

Usage::

    docker compose run --rm estimator python scripts/insert_synthetic_chunks_s08.py        # 100
    docker compose run --rm estimator python scripts/insert_synthetic_chunks_s08.py 30000  # pre-flight
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import tiktoken
from sqlalchemy import select

from s08_common import require_embedder

from app.embedding_pipeline.embedder import OpenAIEmbedder, estimated_cost_usd
from app.embedding_pipeline.schemas import Chunk
from app.persistence.database import get_async_session_factory
from app.persistence.models import ChunkRow, DocumentRow

SYNTHETIC_SOURCE_PATH = "synthetic::live-session-08"
SYNTHETIC_DOCUMENT_TYPE = "synthetic_test"
SYNTHETIC_CHUNK_TYPE = "synthetic"
BATCH_SIZE = 500

TEMPLATES = [
    (
        "Component: {feature} for the {sector} sector. Implementation of {detail} "
        "using {tech}. Complexity: {complexity}. Estimated hours: {hours}."
    ),
    (
        "Backend service for {feature} in a {sector} platform, built with {tech}. "
        "Covers {detail}. Estimated effort: {hours} hours."
    ),
    (
        "Integration module connecting {feature} with external providers for "
        "{sector} clients. Stack: {tech}. Scope includes {detail}."
    ),
    (
        "Data pipeline for {feature} supporting {sector} operations: {detail}. "
        "Technologies: {tech}. Sized at {hours} hours, {complexity} complexity."
    ),
    (
        "Frontend dashboard exposing {feature} to {sector} users. {detail} "
        "rendered with {tech}. Estimated at {hours} hours."
    ),
]

FEATURES = [
    "OAuth 2.0 authentication",
    "payment reconciliation",
    "fraud scoring",
    "inventory forecasting",
    "appointment scheduling",
    "document ingestion",
    "real-time telemetry",
    "notification delivery",
    "customer onboarding",
    "audit logging",
    "report generation",
    "search indexing",
]
SECTORS = ["finance", "ecommerce", "healthcare", "industrial"]
TECHS = [
    "ruby_on_rails and postgresql",
    "python and fastapi",
    "java and spring",
    "node and express",
    "elixir and phoenix",
    "go and grpc",
]
DETAILS = [
    "token lifecycle, refresh flows and rate limiting",
    "batch jobs with retry queues and dead-letter handling",
    "webhook fan-out with signature verification",
    "schema validation and incremental loading",
    "role-based access control and session management",
    "metrics aggregation with alerting thresholds",
]
COMPLEXITIES = ["low", "medium", "high"]


def synthetic_text(rng: random.Random, sequence: int) -> str:
    template = rng.choice(TEMPLATES)
    return (
        template.format(
            feature=rng.choice(FEATURES),
            sector=rng.choice(SECTORS),
            tech=rng.choice(TECHS),
            detail=rng.choice(DETAILS),
            complexity=rng.choice(COMPLEXITIES),
            hours=rng.randint(8, 320),
        )
        + f" [synthetic #{sequence}]"
    )


async def find_or_create_synthetic_document(session) -> int:
    stmt = select(DocumentRow.id).where(DocumentRow.source_path == SYNTHETIC_SOURCE_PATH)
    document_id = (await session.execute(stmt)).scalar_one_or_none()
    if document_id is not None:
        return document_id
    document = DocumentRow(
        source_path=SYNTHETIC_SOURCE_PATH,
        document_type=SYNTHETIC_DOCUMENT_TYPE,
        metadata_json={"source": "live_session_test"},
    )
    session.add(document)
    await session.flush()
    return document.id


async def insert_batch(
    session_factory,
    embedder: OpenAIEmbedder,
    encoding,
    rng: random.Random,
    start: int,
    size: int,
) -> int:
    """Generate, embed and persist one batch in a single transaction."""
    texts = [synthetic_text(rng, start + offset) for offset in range(size)]
    chunks = [
        Chunk(
            chunk_id=f"synthetic::{start + offset}",
            text=body,
            metadata={"source": "live_session_test"},
            token_count=len(encoding.encode(body)),
        )
        for offset, body in enumerate(texts)
    ]
    embedded = embedder.embed_many(chunks)

    async with session_factory() as session, session.begin():
        document_id = await find_or_create_synthetic_document(session)
        session.add_all(
            ChunkRow(
                document_id=document_id,
                chunk_type=SYNTHETIC_CHUNK_TYPE,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_json=chunk.metadata,
            )
            for chunk in embedded
        )
    return sum(chunk.token_count for chunk in embedded)


async def insert_synthetic(count: int) -> None:
    embedder = require_embedder()
    encoding = tiktoken.get_encoding("cl100k_base")
    session_factory = get_async_session_factory()
    rng = random.Random()

    started = time.perf_counter()
    total_tokens = 0
    inserted = 0
    batch_size = 10 if count <= 200 else BATCH_SIZE

    while inserted < count:
        size = min(batch_size, count - inserted)
        total_tokens += await insert_batch(session_factory, embedder, encoding, rng, inserted, size)
        inserted += size
        print(f"  {inserted}/{count} chunks inserted ({time.perf_counter() - started:.0f}s elapsed)")

    elapsed = time.perf_counter() - started
    print()
    print(
        f"Done: {inserted} synthetic chunks inserted in {elapsed:.1f}s — "
        f"{total_tokens} tokens embedded, estimated cost ${estimated_cost_usd(total_tokens):.4f}"
    )
    print("Clean-up: DELETE FROM documents WHERE document_type = 'synthetic_test';")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insert synthetic chunks with real embeddings into the chunks table."
    )
    parser.add_argument(
        "count",
        type=int,
        nargs="?",
        default=100,
        help="Number of chunks to insert (default: 100; use ~30000 for the pre-flight bulk).",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be a positive integer")

    asyncio.run(insert_synthetic(args.count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

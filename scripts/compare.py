#!/usr/bin/env python3
"""Compare two OpenAI text embeddings using only stdlib cosine arithmetic.

Run from the project root, with OPENAI_API_KEY in .env:
    uv run python scripts/compare.py --text-a "OAuth backend" --text-b "JWT service"
Or inside Docker:
    docker compose exec estimator python scripts/compare.py --text-a "A" --text-b "B"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embedding_pipeline.embedder import OpenAIEmbedder  # noqa: E402


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Dot product divided by the product of the L2 norms."""
    dot = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosine similarity between two embedded texts.")
    parser.add_argument("--text-a", required=True, help="First text.")
    parser.add_argument("--text-b", required=True, help="Second text.")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set (check your .env).", file=sys.stderr)
        return 1

    embedder = OpenAIEmbedder(
        client=OpenAI(api_key=settings.OPENAI_API_KEY),
        model=settings.EMBEDDING_MODEL,
    )
    vector_a = embedder.embed_one(args.text_a)
    vector_b = embedder.embed_one(args.text_b)
    similarity = cosine_similarity(vector_a, vector_b)

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
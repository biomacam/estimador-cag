"""Pre-flight check: does the cross-encoder download, load and score?

Run BEFORE enabling reranking::

    docker compose exec estimator python -m app.rag.retrieval.verify_reranker
    .venv\\Scripts\\python.exe -m app.rag.retrieval.verify_reranker

It loads the configured model and scores a tiny pair where one document is
obviously more relevant than the other. Exit code 0 = ready; 1 = the model could
not load or run; 2 = it loaded but ranked the pair the wrong way.
"""

from __future__ import annotations

import sys

from app.rag.retrieval.reranker import CrossEncoderReranker

_QUERY = "e-commerce checkout and shopping cart platform"
_DOCUMENTS = [
    "Online store checkout flow with shopping cart, payment and order management.",
    "Hospital patient appointment scheduling and telemedicine video consultations.",
]


def main() -> int:
    reranker = CrossEncoderReranker.from_settings()
    print(f"Loading reranker model: {reranker.model_name} ...")
    try:
        scores = reranker.score(_QUERY, _DOCUMENTS)
    except Exception as exc:  # noqa: BLE001 — surface any load/score failure.
        print(f"FAILED to load or run the reranker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Query:          {_QUERY!r}")
    print(f"Relevant doc:   score = {scores[0]:.4f}")
    print(f"Irrelevant doc: score = {scores[1]:.4f}")

    if scores[0] <= scores[1]:
        print(
            "WARNING: the relevant document did not outscore the irrelevant one. "
            "The model loaded, but its ranking looks off — check RERANKER_MODEL.",
            file=sys.stderr,
        )
        return 2

    print("OK: reranker loaded and ranked the relevant document first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Augmentation: assemble retrieved chunks into a delimited context block.

Each chunk is wrapped in a ``<source>`` element carrying its id and metadata so
the generator can cite it, and the context is truncated by dropping WHOLE chunks
from the tail (least relevant first) — a half budget line item is worse than one
fewer.
"""

from __future__ import annotations

from app.rag.schemas import RetrievedChunk


def _wrap_chunk(chunk: RetrievedChunk) -> str:
    return (
        f'<source id="{chunk.id}" sector="{chunk.sector}" '
        f'project_year="{chunk.project_year}" chunk_type="{chunk.chunk_type}" '
        f'distance="{chunk.distance:.4f}">\n'
        f"{chunk.content}\n"
        f"</source>"
    )


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Concatenate the chunks (most relevant first) as ``<source>`` elements."""
    return "\n".join(_wrap_chunk(chunk) for chunk in chunks)


def truncate_to_token_budget(
    chunks: list[RetrievedChunk],
    max_context_tokens: int,
    encoder,
) -> list[RetrievedChunk]:
    """Keep the leading chunks that fit in ``max_context_tokens``.

    Tokens are counted over the chunk already wrapped in its XML (the delimiters
    are real tokens). The first chunk that would overflow stops the loop.
    """
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(encoder.encode(_wrap_chunk(chunk)))
        if used + cost > max_context_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept

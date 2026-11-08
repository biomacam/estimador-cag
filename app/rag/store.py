"""Async data-access layer for filtered vector search and lexical search.

The store never opens or commits sessions: the caller owns the ``AsyncSession``.
Rows expose the JSONB column as ``metadata_`` (labelled in the SELECT) because
``metadata`` is reserved on declarative models.
"""

from __future__ import annotations

from sqlalchemy import Integer, Row, cast, column, func, select
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import HALFVEC

from app.embedding_pipeline.embedder import EMBEDDING_DIM
from app.persistence.models import ChunkRow

# Generated column created by migration 0004 (Postgres-only). Deliberately not mapped
# on ``ChunkRow`` so the ORM metadata still builds on SQLite (used by most tests).
_CONTENT_TSV = column("content_tsv", TSVECTOR)


class ChunkStore:
    """Metadata-filtered k-NN and full-text ranking over ``chunks``."""

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> tuple[list[Row], int]:
        """k-NN by cosine distance with structural pre-filtering and a relevance floor.

        Returns ``(rows, candidates_evaluated)``: the top-k chunks under the
        threshold (ascending distance) and how many chunks matched the structural
        filters before the threshold/limit were applied.
        """
        structural_filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        # Cast so Postgres can serve the query from chunks_embedding_halfvec_idx
        # (scripts/sql_s08/03_create_halfvec.sql); without that index it is a plain exact scan.
        distance = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIM)).cosine_distance(query_vector)

        count_stmt = select(func.count()).select_from(ChunkRow).where(*structural_filters)
        candidates_evaluated = int((await session.execute(count_stmt)).scalar_one())

        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_json.label("metadata_"),
                distance.label("distance"),
            )
            .where(ChunkRow.embedding.is_not(None))
            .where(*structural_filters)
            .where(distance <= distance_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        return list((await session.execute(stmt)).all()), candidates_evaluated

    async def search_lexical(
        self,
        session: AsyncSession,
        *,
        query_text: str,
        top_k: int = 50,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[Row]:
        """Keyword ranking over ``content_tsv``: the lexical branch of hybrid search.

        ``plainto_tsquery`` + ``@@`` keeps matching chunks and ``ts_rank_cd`` ranks
        them (higher = better, opposite to vector distance). The ``english`` config
        must match the generated column's config (migration 0004).
        """
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank_cd(_CONTENT_TSV, tsquery)

        structural_filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_json.label("metadata_"),
                rank.label("rank"),
            )
            .where(_CONTENT_TSV.op("@@")(tsquery))
            .where(*structural_filters)
            .order_by(rank.desc())
            .limit(top_k)
        )
        return list((await session.execute(stmt)).all())

    @staticmethod
    def _structural_filters(
        *,
        sectors: list[str] | None,
        project_year_min: int | None,
        project_year_max: int | None,
        chunk_types: list[str] | None,
    ) -> list:
        """Shared predicates; ``None`` means "do not filter on this axis"."""
        sector_col = ChunkRow.metadata_json["client_sector"].astext
        year_col = cast(ChunkRow.metadata_json["year"].astext, Integer)

        filters = []
        if sectors:
            filters.append(sector_col.in_(sectors))
        if project_year_min is not None:
            filters.append(year_col >= project_year_min)
        if project_year_max is not None:
            filters.append(year_col <= project_year_max)
        if chunk_types:
            filters.append(ChunkRow.chunk_type.in_(chunk_types))
        return filters

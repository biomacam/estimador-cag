"""Full-text search column on chunks (lexical branch of hybrid search).

Adds a STORED generated ``content_tsv`` derived from ``content`` plus a GIN index,
so the retriever can run a keyword (``ts_rank_cd``) branch next to the vector
branch and fuse both with RRF. The ``english`` configuration matches the corpus
language (stemming and stop-words); it must stay identical to the
``plainto_tsquery`` call in ``app.rag.store`` or the GIN index is bypassed.

GENERATED ALWAYS ... STORED: Postgres recomputes the tsvector on every write of
``content``, so there is no trigger to maintain and no drift.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0004_chunks_fts"
down_revision: Union[str, None] = "0003_unique_document_source_path"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Raw DDL: SQLAlchemy has no first-class helper for a STORED generated tsvector.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.create_index("ix_chunks_content_tsv", "chunks", ["content_tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_chunks_content_tsv", table_name="chunks")
    op.drop_column("chunks", "content_tsv")

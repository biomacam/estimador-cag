"""Make document source paths unique for idempotent ingestion."""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003_unique_document_source_path"
down_revision: Union[str, None] = "0002_document_chunks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_documents_source_path", table_name="documents")
    op.create_index("ix_documents_source_path", "documents", ["source_path"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_documents_source_path", table_name="documents")
    op.create_index("ix_documents_source_path", "documents", ["source_path"])
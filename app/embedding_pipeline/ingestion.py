"""Transactional persistence for structurally chunked source documents."""

from __future__ import annotations

import asyncio
import time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import EMBEDDING_DIM, OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse
from app.persistence.models import ChunkRow, DocumentRow


class DuplicateDocumentError(Exception):
    def __init__(self, document_id: int) -> None:
        self.document_id = document_id
        super().__init__("Document already ingested")


async def ingest_document(
    request: IngestRequest,
    session: AsyncSession,
    chunker: JSONStructuralChunker,
    embedder: OpenAIEmbedder,
) -> IngestResponse:
    """Chunk, embed and persist one source document in a single transaction."""
    started = time.perf_counter()
    try:
        async with session.begin():
            existing_id = await session.scalar(
                select(DocumentRow.id).where(DocumentRow.source_path == request.source_path)
            )
            if existing_id is not None:
                raise DuplicateDocumentError(existing_id)

            document = DocumentRow(
                source_path=request.source_path,
                document_type=request.document_type,
                metadata_json=request.content.model_dump(
                    mode="json", exclude={"components"}
                ),
            )
            session.add(document)
            await session.flush()

            chunks = chunker.chunk([request.content])
            embedded_chunks = await asyncio.to_thread(embedder.embed_all, chunks)
            if len(embedded_chunks) != len(chunks):
                raise RuntimeError("Embedding response count does not match chunk count.")

            session.add_all(
                [
                    ChunkRow(
                        document_id=document.id,
                        chunk_type="budget_component",
                        content=chunk.text,
                        embedding=embedded.embedding,
                        metadata_json=chunk.metadata,
                    )
                    for chunk, embedded in zip(chunks, embedded_chunks, strict=True)
                ]
            )
            document_id = document.id
            chunks_created = len(chunks)
    except IntegrityError as exc:
        existing_id = await session.scalar(
            select(DocumentRow.id).where(DocumentRow.source_path == request.source_path)
        )
        if existing_id is not None:
            raise DuplicateDocumentError(existing_id) from exc
        raise

    return IngestResponse(
        document_id=document_id,
        chunks_created=chunks_created,
        embedding_dimension=EMBEDDING_DIM,
        ingestion_time_ms=round((time.perf_counter() - started) * 1000),
    )
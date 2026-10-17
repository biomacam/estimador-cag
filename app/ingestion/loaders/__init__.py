"""Loaders: raw byte producers.

A loader enumerates files (or URLs, or DB rows) and hands the parser a stream of
blobs. Loaders never look INTO the bytes; that is the parser's job.
"""

from app.ingestion.loaders.filesystem import FileSystemLoader, LoadedBlob

__all__ = ["FileSystemLoader", "LoadedBlob"]

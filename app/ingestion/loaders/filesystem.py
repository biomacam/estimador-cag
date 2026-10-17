"""Filesystem loader.

Resolves ``CatalogSource.location`` against ``INGESTION_DATA_ROOT``, walks the
resulting path and yields one ``LoadedBlob`` per file whose extension matches the
source's declared format. A location that does not exist is a hard error: a
catalog that points nowhere is silently wrong. A location (or file) that
resolves outside the data root is rejected too, since the catalog is data and
must not be able to read arbitrary files.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedBlob:
    """A single file's bytes plus its path relative to the data root."""

    relative_path: str
    bytes_: bytes


class FileSystemLoader:
    def __init__(self, data_root: Path | str) -> None:
        self._data_root = Path(data_root).resolve()

    def iter_blobs(self, location: str, formats: set[str]) -> Iterator[LoadedBlob]:
        path = self._resolve_inside_root(self._data_root / location, location)
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog location {location!r} does not resolve to a real path ({path!s} not found)"
            )
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            ext = candidate.suffix.lower().lstrip(".")
            if ext not in formats:
                continue
            # Resolving each file also catches symlinks that point outside the root.
            resolved = self._resolve_inside_root(candidate, location)
            yield LoadedBlob(
                relative_path=resolved.relative_to(self._data_root).as_posix(),
                bytes_=resolved.read_bytes(),
            )

    def _resolve_inside_root(self, path: Path, location: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self._data_root):
            raise ValueError(f"Catalog location {location!r} escapes the ingestion data root")
        return resolved

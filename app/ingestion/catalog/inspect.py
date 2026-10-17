"""Facts-only inspection of a raw data folder.

Produces *facts* (file_count, total_size_mb, latest_modified, formats_detected)
that ground the catalog entries. It deliberately does NOT score quality, write
the catalog or read it back: opinions live in YAML, facts live in code.

Run from the project root::

    python -m app.ingestion.catalog.inspect data/seed
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FolderFacts:
    folder: Path
    file_count: int = 0
    total_size_mb: float = 0.0
    latest_modified: datetime | None = None
    formats_detected: set[str] = field(default_factory=set)

    def add_file(self, path: Path) -> None:
        stat = path.stat()
        self.file_count += 1
        self.total_size_mb += stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        if self.latest_modified is None or modified > self.latest_modified:
            self.latest_modified = modified
        if path.suffix:
            self.formats_detected.add(path.suffix.lower().lstrip("."))

    def as_table_row(self) -> str:
        ts = self.latest_modified.isoformat() if self.latest_modified else "-"
        formats = ",".join(sorted(self.formats_detected)) or "-"
        return (
            f"{self.folder.name:<20} "
            f"{self.file_count:>5} files  "
            f"{self.total_size_mb:>7.2f} MB  "
            f"latest={ts:<25} "
            f"formats={formats}"
        )


def inspect_folder(folder: Path) -> FolderFacts:
    facts = FolderFacts(folder=folder)
    if not folder.exists():
        return facts
    for path in folder.rglob("*"):
        if path.is_file():
            facts.add_file(path)
    return facts


def inspect_root(root: Path) -> list[FolderFacts]:
    """Inspect each immediate child folder of ``root`` plus the files directly under it."""
    results: list[FolderFacts] = []
    if not root.exists():
        return results
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        results.append(inspect_folder(child))
    # Files directly under root (e.g. a rate card XLSX) are reported as one group.
    direct_files = [p for p in root.iterdir() if p.is_file()]
    if direct_files:
        bag = FolderFacts(folder=root)
        for path in direct_files:
            bag.add_file(path)
        results.append(bag)
    return results


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m app.ingestion.catalog.inspect <root-folder>")
        return 1
    root = Path(argv[1])
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return 2
    facts = inspect_root(root)
    if not facts:
        print(f"(no files under {root})")
        return 0
    print(f"Inspection of {root.resolve()}\n")
    for entry in facts:
        print(entry.as_table_row())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

#!/usr/bin/env python3
"""Report the state of every index on the ``chunks`` table.

Session 08 — run before and after every index decision (create, drop,
reindex). Sources: ``pg_stat_user_indexes`` (usage), ``pg_relation_size``
(size) and ``pg_am`` (access method: btree / gin / hnsw). Read-only.

Usage::

    docker compose run --rm estimator python scripts/report_index_sizes_s08.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.persistence.database import get_async_session_factory  # noqa: E402

REPORT_SQL = text(
    """
    SELECT
        i.indexrelname AS index_name,
        am.amname AS index_type,
        pg_size_pretty(pg_relation_size(i.indexrelid)) AS size,
        i.idx_scan AS scans,
        COALESCE(i.last_idx_scan::text, 'never') AS last_used
    FROM pg_stat_user_indexes i
    JOIN pg_class c ON c.oid = i.indexrelid
    JOIN pg_am am ON am.oid = c.relam
    WHERE i.relname = 'chunks'
    ORDER BY pg_relation_size(i.indexrelid) DESC
    """
)

TABLE_SQL = text(
    """
    SELECT
        pg_size_pretty(pg_table_size('chunks')) AS table_size,
        (SELECT count(*) FROM chunks) AS chunk_count
    """
)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in rows)
    return "\n".join(lines)


async def report() -> None:
    async with get_async_session_factory()() as session:
        table_info = (await session.execute(TABLE_SQL)).one()
        index_rows = (await session.execute(REPORT_SQL)).all()

    print(f"chunks table: {table_info.chunk_count} rows, {table_info.table_size}")
    print()
    if not index_rows:
        print("No indexes found on chunks — has the migration run?")
        return
    print(
        format_table(
            ["index_name", "type", "size", "scans", "last_used"],
            [
                [row.index_name, row.index_type, row.size, str(row.scans), row.last_used]
                for row in index_rows
            ],
        )
    )
    print()
    print(
        "An index with scans = 0 after running semantic queries usually means "
        "the operator in the query does not match the index operator class."
    )


def main() -> int:
    asyncio.run(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

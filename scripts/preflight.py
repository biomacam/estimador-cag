"""Pre-flight verification for the ingestion stack.

Run it before a demo to catch the usual setup pitfalls: missing Spanish spaCy model,
broken catalog, incomplete seed corpus, Postgres unreachable or unmigrated. Prints
OK / FAIL per check and exits non-zero if anything fails.

Usage::

    uv run python scripts/preflight.py
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

OK = "[OK]  "
FAIL = "[FAIL]"

ROOT = Path(__file__).resolve().parent.parent
# Make ``app`` importable when this script runs directly from ``scripts/``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Check:
    def __init__(self, name: str, fn: Callable[[], str | None]) -> None:
        self.name = name
        self._fn = fn

    def run(self) -> bool:
        try:
            detail = self._fn()
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} {self.name}: {type(exc).__name__}: {exc}")
            return False
        print(f"{OK} {self.name}" + (f": {detail}" if detail else ""))
        return True


def check_python_version() -> str:
    major, minor, *_ = sys.version_info
    if (major, minor) < (3, 14):
        raise RuntimeError(f"Python 3.14+ required, found {major}.{minor}")
    return f"{major}.{minor}.{sys.version_info.micro}"


_MIN_VERSIONS = {
    "pandas": "2.2.0",
    "pandera": "0.20.0",
    "pyyaml": "6.0",
    "sqlalchemy": "2.0.0",
    "psycopg": "3.1.0",
    "alembic": "1.13.0",
    "openpyxl": "3.1.0",
    "presidio-analyzer": "2.2.0",
    "presidio-anonymizer": "2.2.0",
    "spacy": "3.8.0",
    "faker": "24.0.0",
}


def _parse_version(s: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in s.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def check_packages() -> str:
    missing: list[str] = []
    too_old: list[str] = []
    for pkg, minimum in _MIN_VERSIONS.items():
        try:
            installed = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            missing.append(pkg)
            continue
        if _parse_version(installed) < _parse_version(minimum):
            too_old.append(f"{pkg} {installed} < {minimum}")
    if missing or too_old:
        raise RuntimeError("missing=" + ",".join(missing) + " too_old=" + ",".join(too_old))
    return f"{len(_MIN_VERSIONS)} packages OK"


def check_spacy_model() -> str:
    import spacy

    from app.config import get_settings

    model = get_settings().PRESIDIO_SPACY_MODEL
    doc = spacy.load(model)("Laura Fernández firmó el contrato.")
    persons = [e.text for e in doc.ents if e.label_ == "PER"]
    if "Laura Fernández" not in " ".join(persons):
        raise RuntimeError(
            f"{model} loaded but did NOT detect 'Laura Fernández' as PER; "
            "the model may be too small or corrupted"
        )
    return f"{model} loaded; detected PER entities: {persons}"


def check_corpus_seed() -> str:
    seed = ROOT / "data" / "seed"
    budgets = list((seed / "budgets").glob("*.json"))
    transcripts = list((seed / "transcripts").glob("*.txt"))
    xlsx = list(seed.glob("*.xlsx"))
    if not budgets or not transcripts or not xlsx:
        raise RuntimeError(
            f"corpus incomplete: budgets={len(budgets)}, "
            f"transcripts={len(transcripts)}, xlsx={len(xlsx)}"
        )
    return f"{len(budgets)} json + {len(transcripts)} txt + {len(xlsx)} xlsx"


def check_catalog() -> str:
    from app.ingestion.catalog import load_catalog

    catalog = load_catalog(ROOT / "data" / "catalog" / "catalog.yaml")
    return f"version {catalog.version}, {len(catalog.included_sources())} included"


def check_health_endpoint() -> str:
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=2) as r:
            return f"{r.status} OK"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GET /health failed; is the estimator running? ({exc})") from exc


def check_postgres() -> str:
    """Connect to Postgres and confirm the migration has been applied."""
    from sqlalchemy import create_engine, inspect

    from app.config import get_settings

    # Own engine: psycopg has no connect timeout by default, so an unreachable host would hang.
    engine = create_engine(get_settings().DATABASE_URL, connect_args={"connect_timeout": 3})
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    expected = {"pseudonym_mappings", "ingestion_jobs", "alembic_version"}
    missing = expected - tables
    if missing:
        raise RuntimeError(f"missing tables: {sorted(missing)}; run `alembic upgrade head`")
    return f"tables OK: {sorted(expected)}"


CHECKS = [
    Check("Python version", check_python_version),
    Check("Required packages", check_packages),
    Check("spaCy Spanish model", check_spacy_model),
    Check("Corpus seed present", check_corpus_seed),
    Check("Catalog validates", check_catalog),
    Check("Estimator /health", check_health_endpoint),
    Check("Postgres + migration", check_postgres),
]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    print("Pre-flight: ingestion stack\n")
    failed = sum(not check.run() for check in CHECKS)
    print()
    if failed:
        print(f"{FAIL} {failed} check(s) failed")
        return 1
    print(f"{OK} All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

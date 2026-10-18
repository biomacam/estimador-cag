"""Demo: pseudonymization with Presidio (Spanish) and a mapping store.

Builds ``build_analyzer()`` (``es_core_news_md`` plus the custom recognizers), runs a
``ConsistentPseudonymizer`` over a seed transcript and prints the result. It does not
run the "default English engine fails" contrast: that would trigger a ~382 MB
``en_core_web_lg`` download, and inside Docker only the Spanish model is installed.

Usage::

    uv run python scripts/demo_pii.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A salt is required to build the pseudonymizer; use a demo one if unset.
os.environ.setdefault("PSEUDONYM_HASH_SALT", "demo-salt")

from app.ingestion.pii import (  # noqa: E402
    ConsistentPseudonymizer,
    InMemoryMappingStore,
    build_analyzer,
)

TRANSCRIPT = ROOT / "data" / "seed" / "transcripts" / "transcripcion_2025-02-03_betanorte.txt"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    if not TRANSCRIPT.exists():
        print(f"ERROR: {TRANSCRIPT} does not exist")
        return 1

    pseudo = ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=InMemoryMappingStore(),
        salt=os.environ["PSEUDONYM_HASH_SALT"],
    )
    result = pseudo.pseudonymize(TRANSCRIPT.read_text(encoding="utf-8"))

    print(f"Transcript: {TRANSCRIPT.name}")
    print("\n--- Pseudonymized text (first 400 chars) ---")
    print(result.pseudonymized_text[:400])

    print(f"\n--- Mappings applied ({len(result.applied)}) ---")
    for m in result.applied[:12]:
        print(f"  {m.entity_type:14} -> {m.pseudonym}")
    print("\nThe same original value gets the same pseudonym across the whole document.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

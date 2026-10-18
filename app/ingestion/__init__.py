"""Data ingestion subsystem.

Three conceptual layers, each in its own subpackage:

* ``catalog`` - versioned audit of data sources (what we ingest and why).
* ``loaders`` + ``parsers`` - raw bytes -> list[Document] (the canonical contract).
* ``cleaning`` + ``pii`` - tabular validation and GDPR pseudonymization.

The HTTP entry point lives in ``app.routers.ingestion``; the glue that ties
everything together lives in ``app.ingestion.orchestrator``.
"""

"""Evaluation harness for the conversational estimator.

- ``dataset.py``: typed ``GoldenCase`` records and a JSON loader.
- ``metrics.py``: deterministic metrics (schema adherence, cost bounds, content recall).
- ``run.py``: CLI that fires each case at one of the two modes (actor / acb) and
  prints a per-case, per-metric table.

``golden_dataset.json`` carries 15+ cases mixing project types, scope levels,
NDA/regulatory flavours and a couple of out-of-scope adversarials.
"""

from evals.dataset import GoldenCase, load_dataset
from evals.metrics import (
    ContentRecallMetric,
    CostBoundsMetric,
    MetricResult,
    SchemaAdherenceMetric,
    run_all_metrics,
)

__all__ = [
    "ContentRecallMetric",
    "CostBoundsMetric",
    "GoldenCase",
    "MetricResult",
    "SchemaAdherenceMetric",
    "load_dataset",
    "run_all_metrics",
]

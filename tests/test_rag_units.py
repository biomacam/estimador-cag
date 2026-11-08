"""Unit tests for the pure pieces of the RAG pipeline: validation, context assembly,
prompt building, rank fusion and the lexical/filtered SQL."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql

from app.rag.context_assembler import build_context_block, truncate_to_token_budget
from app.rag.prompt_builder import build_system_prompt, build_user_message
from app.rag.query_reformulator import compose_search_text
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.reranker import CrossEncoderReranker
from app.rag.schemas import (
    Estimate,
    EstimationQuery,
    RetrievedChunk,
    SourceCitation,
    TaskItem,
    WorkModule,
)
from app.rag.validation import check_coherence, validate_citations


class CharEncoder:
    """One token per character."""

    def encode(self, text: str) -> list[str]:
        return list(text)


def _chunk(chunk_id: int, content: str = "Component: Auth\nEstimated hours: 120") -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=content,
        sector="ecommerce",
        project_year=2024,
        chunk_type="budget_component",
        distance=0.123,
    )


def _estimate(*, source_ids, task_sources, confidence="high") -> Estimate:
    return Estimate(
        total_engineer_days=20,
        duration_weeks=4,
        modules=[
            WorkModule(
                name="Authentication",
                tasks=[TaskItem(name="Auth", engineer_days=20, sources=task_sources)],
            )
        ],
        sources=[
            SourceCitation(source_id=sid, relevance="primary", used_for="auth")
            for sid in source_ids
        ],
        confidence=confidence,
        reasoning="Derived from retrieved budgets.",
    )


# --- validation ------------------------------------------------------------


def test_validate_citations_all_valid():
    assert validate_citations(_estimate(source_ids=[1, 2], task_sources=[1]), [_chunk(1), _chunk(2)]) == []


def test_validate_citations_flags_top_level_and_task_level_fabrications():
    estimate = _estimate(source_ids=[1, 99], task_sources=[42])
    assert validate_citations(estimate, [_chunk(1), _chunk(2)]) == [42, 99]


def test_validate_citations_empty_retrieval_flags_everything():
    assert validate_citations(_estimate(source_ids=[1], task_sources=[2]), []) == [1, 2]


def test_check_coherence_insufficient_shape():
    ok = Estimate(
        confidence="insufficient",
        reasoning="r",
        insufficient_context_explanation="No relevant budgets retrieved.",
    )
    with_numbers = Estimate(
        confidence="insufficient",
        reasoning="r",
        total_engineer_days=10,
        insufficient_context_explanation="x",
    )
    assert check_coherence(ok) is True
    assert check_coherence(with_numbers) is False
    assert check_coherence(_estimate(source_ids=[], task_sources=[], confidence="low")) is True


# --- context assembler -----------------------------------------------------


def test_build_context_block_wraps_chunks_in_order():
    block = build_context_block([_chunk(1, "first"), _chunk(2, "second")])

    assert '<source id="1" sector="ecommerce" project_year="2024"' in block
    assert 'distance="0.1230"' in block
    assert block.index('id="1"') < block.index('id="2"')
    assert block.strip().endswith("</source>")
    assert build_context_block([]) == ""


def test_truncate_keeps_whole_chunks_and_counts_the_xml_wrapper():
    encoder = CharEncoder()
    chunks = [_chunk(1, "a" * 50), _chunk(2, "b" * 50), _chunk(3, "c" * 50)]
    first_two = sum(len(build_context_block([c])) for c in chunks[:2])

    kept = truncate_to_token_budget(chunks, first_two, encoder)

    assert [c.id for c in kept] == [1, 2]
    assert truncate_to_token_budget([_chunk(1, "x")], 1, encoder) == []  # wrapper does not fit


# --- prompts ---------------------------------------------------------------


def test_prompts_carry_grounding_rules_and_the_brief():
    assert 'confidence="insufficient"' in build_system_prompt()
    message = build_user_message(
        "<source id=\"1\">x</source>",
        EstimationQuery(function="online store", technologies=["Rails"], sector="ecommerce"),
    )
    assert "Function: online store" in message
    assert "Technologies: Rails" in message
    assert '<sources>\n<source id="1">x</source>\n</sources>' in message


def test_compose_search_text_joins_grounded_fields_and_drops_empty_ones():
    query = EstimationQuery(
        function="B2B payments marketplace",
        technologies=["Stripe Connect", "KYC"],
        sector="finance",
        country="Germany",
        regulations=["BaFin"],
    )
    text = compose_search_text(query)

    assert text == (
        "B2B payments marketplace with Stripe Connect, KYC for finance in Germany BaFin-compliant"
    )
    assert compose_search_text(EstimationQuery(function="internal tool")) == "internal tool"


# --- reciprocal rank fusion ------------------------------------------------


def test_rrf_rewards_agreement_between_branches():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]])

    assert [cid for cid, _ in fused][:2] == [1, 3]  # present in both branches
    assert {cid for cid, _ in fused} == {1, 2, 3}


def test_rrf_ignores_duplicates_empty_branches_and_breaks_ties_by_id():
    assert reciprocal_rank_fusion([[5, 5, 5], []]) == [(5, pytest.approx(1 / 60))]
    assert [cid for cid, _ in reciprocal_rank_fusion([[2], [1]])] == [1, 2]


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=0)


# --- reranker --------------------------------------------------------------


class _FakeModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, pairs):
        return [self._scores[document] for _query, document in pairs]


def test_reranker_orders_by_score_and_cuts_to_top_n():
    reranker = CrossEncoderReranker("fake-model")
    reranker._model = _FakeModel({"a": 0.1, "b": 0.9, "c": 0.5})  # skip lazy loading

    ranked = reranker.rerank("q", ["a", "b", "c"], top_n=2, text_of=lambda doc: doc)

    assert ranked == ["b", "c"]
    assert reranker.rerank("q", [], top_n=3) == []


# --- SQL built by the store ------------------------------------------------


def test_lexical_and_filtered_statements_compile_to_expected_sql(monkeypatch):
    import asyncio

    from app.rag.store import ChunkStore

    captured = []

    class _Result:
        def all(self):
            return []

        def scalar_one(self):
            return 0

    class _Session:
        async def execute(self, stmt):
            captured.append(stmt)
            return _Result()

    store = ChunkStore()

    async def scenario():
        await store.search_filtered(
            _Session(), query_vector=[0.0] * 3, sectors=["finance"], project_year_min=2022
        )
        await store.search_lexical(_Session(), query_text="oauth login", chunk_types=["budget_component"])

    asyncio.run(scenario())
    sql = [str(s.compile(dialect=postgresql.dialect())) for s in captured]

    count_sql, vector_sql, lexical_sql = sql
    assert "count(*)" in count_sql
    assert "<=>" in vector_sql and "->>" in vector_sql and "IN" in vector_sql  # JSONB sector filter
    assert "halfvec" in vector_sql.lower()  # served by the halfvec expression index
    assert "plainto_tsquery" in lexical_sql and "@@" in lexical_sql and "ts_rank_cd" in lexical_sql
    assert "chunk_type IN" in lexical_sql

from app.attachments import extract_text
from app.config import Settings
from evals.stress.fixtures.build_pdfs import TARGETS_KB, build_all, pdf_path


def test_pdfs_are_calibrated_deterministic_and_truncated_at_the_cap(tmp_path):
    paths = build_all(tmp_path)
    first = {p.name: p.read_bytes() for p in paths}

    assert [p.name for p in paths] == [f"attach_{kb}kb.pdf" for kb in TARGETS_KB]

    cap = Settings().MAX_ATTACHMENT_CHARS
    for kb in TARGETS_KB:
        content = pdf_path(kb, tmp_path).read_bytes()
        full = extract_text(f"attach_{kb}kb.pdf", content, max_chars=10**9)
        assert kb * 1024 * 0.9 <= len(full) <= kb * 1024 * 1.1
        capped = extract_text(f"attach_{kb}kb.pdf", content, max_chars=cap)
        assert ("truncated" in capped) == (len(full) > cap)

    build_all(tmp_path)
    assert {p.name: p.read_bytes() for p in paths} == first

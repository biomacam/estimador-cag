import pytest

from app.attachments import (
    AttachmentExtractionError,
    UnsupportedAttachmentError,
    enrich_transcript,
    extract_text,
    safe_filename,
)
from tests.helpers import make_docx, make_pdf


def test_extracts_text_from_a_pdf() -> None:
    text = extract_text("spec.pdf", make_pdf("Budget for the CRM project"), max_chars=1000)

    assert "Budget for the CRM project" in text


def test_extracts_paragraphs_and_table_cells_from_a_docx() -> None:
    content = make_docx(["Scope of the portal"], table_rows=[["Module", "Weeks"], ["Billing", "6"]])

    text = extract_text("notes.docx", content, max_chars=1000)

    assert "Scope of the portal" in text
    assert "Billing | 6" in text


def test_extension_check_is_case_insensitive() -> None:
    assert "Upper case name" in extract_text("SPEC.PDF", make_pdf("Upper case name"), max_chars=1000)


@pytest.mark.parametrize("filename", ["notes.txt", "legacy.doc", "noextension"])
def test_unsupported_extensions_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedAttachmentError):
        extract_text(filename, b"plain text", max_chars=1000)


@pytest.mark.parametrize("filename", ["broken.pdf", "broken.docx"])
def test_corrupt_files_raise_an_extraction_error(filename: str) -> None:
    with pytest.raises(AttachmentExtractionError) as excinfo:
        extract_text(filename, b"this is not a real document", max_chars=1000)

    assert excinfo.value.filename == filename


def test_document_without_text_raises_an_extraction_error() -> None:
    with pytest.raises(AttachmentExtractionError, match="No extractable text"):
        extract_text("empty.docx", make_docx([]), max_chars=1000)


def test_long_text_is_truncated_with_a_visible_marker() -> None:
    text = extract_text("big.docx", make_docx(["x" * 500]), max_chars=100)

    assert text.startswith("x" * 100)
    assert text.endswith("[attachment truncated at 100 characters]")


def test_enrich_transcript_appends_each_attachment_under_a_separator() -> None:
    enriched = enrich_transcript(
        "Kickoff notes", [("spec.pdf", "PDF body"), ("notes.docx", "DOCX body")]
    )

    assert enriched == (
        "Kickoff notes\n\n"
        "--- attachment: spec.pdf ---\nPDF body\n\n"
        "--- attachment: notes.docx ---\nDOCX body"
    )


def test_enrich_transcript_without_attachments_is_the_transcript() -> None:
    assert enrich_transcript("Kickoff notes", []) == "Kickoff notes"


def test_safe_filename_strips_paths_and_control_characters() -> None:
    assert safe_filename("..\\..\\etc/secret\n--- attachment: fake.pdf") == "secret--- attachment: fake.pdf"
    assert safe_filename("") == "attachment"

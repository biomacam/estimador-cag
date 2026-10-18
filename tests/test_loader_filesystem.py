"""Tests for the filesystem loader, including containment inside the data root."""

from __future__ import annotations

import pytest

from app.ingestion.loaders import FileSystemLoader


def test_yields_only_files_with_requested_formats(tmp_path) -> None:
    (tmp_path / "budgets").mkdir()
    (tmp_path / "budgets" / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "budgets" / "b.JSON").write_text("{}", encoding="utf-8")
    (tmp_path / "budgets" / "notes.txt").write_text("x", encoding="utf-8")

    blobs = list(FileSystemLoader(tmp_path).iter_blobs("budgets", {"json"}))

    assert [b.relative_path for b in blobs] == ["budgets/a.json", "budgets/b.JSON"]
    assert blobs[0].bytes_ == b"{}"


def test_location_can_be_a_single_file(tmp_path) -> None:
    (tmp_path / "rates.json").write_text("{}", encoding="utf-8")

    blobs = list(FileSystemLoader(tmp_path).iter_blobs("rates.json", {"json"}))

    assert [b.relative_path for b in blobs] == ["rates.json"]


def test_walks_subfolders_in_sorted_order(tmp_path) -> None:
    (tmp_path / "t" / "sub").mkdir(parents=True)
    (tmp_path / "t" / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "t" / "sub" / "a.txt").write_text("a", encoding="utf-8")

    blobs = list(FileSystemLoader(tmp_path).iter_blobs("t", {"txt"}))

    assert [b.relative_path for b in blobs] == ["t/sub/a.txt", "t/z.txt"]


def test_missing_location_is_a_hard_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="nope"):
        list(FileSystemLoader(tmp_path).iter_blobs("nope", {"json"}))


@pytest.mark.parametrize("location", ["../outside", "..", "budgets/../../outside"])
def test_location_escaping_the_root_is_rejected(tmp_path, location) -> None:
    root = tmp_path / "root"
    (root / "budgets").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        list(FileSystemLoader(root).iter_blobs(location, {"json"}))


def test_absolute_location_outside_the_root_is_rejected(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="escapes"):
        list(FileSystemLoader(root).iter_blobs(str(outside), {"json"}))


def test_symlink_pointing_outside_the_root_is_rejected(tmp_path) -> None:
    root = tmp_path / "root"
    (root / "budgets").mkdir(parents=True)
    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")
    try:
        (root / "budgets" / "link.json").symlink_to(secret)
    except OSError:
        pytest.skip("symlinks need extra privileges on this platform")

    with pytest.raises(ValueError, match="escapes"):
        list(FileSystemLoader(root).iter_blobs("budgets", {"json"}))

from __future__ import annotations

import importlib
import sys

import numpy as np

from src.vos.cutie_optional import cutie_object_ids_from_indexed, install_or_check_cutie, sanitize_cutie_prediction


def test_cutie_object_ids_from_indexed_keeps_positive_ids_sorted() -> None:
    indexed = np.array(
        [
            [0, 2, 2],
            [5, 0, 1],
        ],
        dtype=np.uint8,
    )

    assert cutie_object_ids_from_indexed(indexed) == [1, 2, 5]


def test_cutie_object_ids_rejects_values_outside_indexed_png_range() -> None:
    indexed = np.array([[0, 256]], dtype=np.uint16)

    try:
        cutie_object_ids_from_indexed(indexed)
    except ValueError as exc:
        assert "8-bit indexed object ids" in str(exc)
    else:
        raise AssertionError("Expected ValueError for object id > 255")


def test_sanitize_cutie_prediction_clears_unexpected_ids() -> None:
    indexed = np.array(
        [
            [0, 1, 2],
            [3, 4, 1],
        ],
        dtype=np.uint8,
    )

    sanitized, unexpected = sanitize_cutie_prediction(indexed, [1, 3])

    assert unexpected == [2, 4]
    assert sanitized.tolist() == [[0, 1, 0], [3, 0, 1]]


def test_install_or_check_cutie_adds_repo_to_current_sys_path(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "Cutie"
    package = repo / "cutie"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    def fake_check_call(cmd: list[str]) -> None:
        assert cmd[:3] == [sys.executable, "-m", "pip"]

    monkeypatch.setattr("subprocess.check_call", fake_check_call)
    sys.modules.pop("cutie", None)
    original_sys_path = list(sys.path)
    sys.path[:] = [path for path in sys.path if str(repo) != path]
    importlib.invalidate_caches()

    try:
        status = install_or_check_cutie(repo, install=True)
        assert status.available is True
        assert str(repo) in sys.path
    finally:
        sys.modules.pop("cutie", None)
        sys.path[:] = original_sys_path
        importlib.invalidate_caches()

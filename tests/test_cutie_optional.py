from __future__ import annotations

import numpy as np

from src.vos.cutie_optional import cutie_object_ids_from_indexed, sanitize_cutie_prediction


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

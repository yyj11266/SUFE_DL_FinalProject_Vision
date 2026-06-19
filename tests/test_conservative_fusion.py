from __future__ import annotations

import numpy as np

from src.vos.conservative_fusion import (
    ConservativeFusionConfig,
    compose_object_masks,
    fuse_frame,
    object_ids_from_indexed,
)


def test_object_ids_from_indexed_returns_positive_sorted_ids() -> None:
    indexed = np.array([[0, 7, 2], [2, 0, 7]], dtype=np.uint8)

    assert object_ids_from_indexed(indexed) == [2, 7]


def test_fuse_frame_preserves_first_frame_exactly() -> None:
    first = np.array([[0, 1, 1], [0, 2, 2]], dtype=np.uint8)
    sam2 = np.zeros_like(first)
    cutie = np.full_like(first, 9)

    result = fuse_frame(sam2, cutie, [1, 2], frame_index=0, first_frame_mask=first)

    assert np.array_equal(result.indexed_mask, first)
    assert {decision.source for decision in result.decisions} == {"prompt"}


def test_fuse_frame_falls_back_to_sam2_when_cutie_empty() -> None:
    sam2 = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint8)
    cutie = np.zeros_like(sam2)

    result = fuse_frame(sam2, cutie, [1], frame_index=1)

    assert np.array_equal(result.indexed_mask, sam2)
    assert result.decisions[0].source == "sam2"
    assert result.decisions[0].reason == "cutie_empty_or_too_small"


def test_fuse_frame_uses_cutie_when_sam2_empty_and_cutie_has_area() -> None:
    sam2 = np.zeros((3, 3), dtype=np.uint8)
    cutie = np.zeros_like(sam2)
    cutie[:2, :2] = 1

    result = fuse_frame(
        sam2,
        cutie,
        [1],
        frame_index=1,
        config=ConservativeFusionConfig(min_cutie_area=1),
    )

    assert np.array_equal(result.indexed_mask, cutie)
    assert result.decisions[0].source == "cutie"
    assert result.decisions[0].reason == "cutie_replaces_empty_sam2"


def test_fuse_frame_uses_cutie_when_global_gates_pass() -> None:
    previous = np.zeros((4, 4), dtype=np.uint8)
    previous[:3, :3] = 1
    sam2 = previous.copy()
    cutie = previous.copy()
    cutie[3, 3] = 1

    result = fuse_frame(
        sam2,
        cutie,
        [1],
        frame_index=2,
        previous_output=previous,
        config=ConservativeFusionConfig(min_cutie_area=1, min_sam2_iou=0.5, min_temporal_iou=0.5),
    )

    assert np.array_equal(result.indexed_mask, cutie)
    assert result.decisions[0].source == "cutie"
    assert result.decisions[0].reason == "cutie_passed_conservative_gates"


def test_fuse_frame_clears_unknown_cutie_ids_before_composition() -> None:
    sam2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    cutie = np.array([[9, 1], [9, 0]], dtype=np.uint8)

    result = fuse_frame(
        sam2,
        cutie,
        [1],
        frame_index=1,
        config=ConservativeFusionConfig(min_cutie_area=1),
    )

    assert 9 not in np.unique(result.indexed_mask).tolist()
    assert result.warnings == ["cutie_unknown_ids_cleared:[9]"]


def test_compose_object_masks_does_not_create_new_ids_on_overlap() -> None:
    masks = {
        1: np.array([[True, True], [False, False]]),
        2: np.array([[True, False], [False, True]]),
    }

    output = compose_object_masks(masks, [1, 2])

    assert output.tolist() == [[1, 1], [0, 2]]

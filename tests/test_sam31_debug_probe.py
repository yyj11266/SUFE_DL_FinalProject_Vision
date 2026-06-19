from __future__ import annotations

from scripts.debug_sam31_api import _mask_probe_summary


def test_mask_probe_summary_accepts_complete_object_ids() -> None:
    summary = _mask_probe_summary(
        {
            "status": "done",
            "object_ids": [1, 2],
            "propagation": [
                {"frame_index": 0, "raw_output_object_ids": [1, 2]},
                {"frame_index": 1, "raw_output_object_ids": [1, 2]},
            ],
        }
    )

    assert summary == {
        "status": "done",
        "maintained_expected_object_ids": True,
        "first_missing_frame": None,
        "empty_non_initial_frames": 0,
    }


def test_mask_probe_summary_reports_early_object_collapse() -> None:
    summary = _mask_probe_summary(
        {
            "status": "done",
            "object_ids": [1, 2],
            "propagation": [
                {"frame_index": 0, "raw_output_object_ids": [1, 2]},
                {"frame_index": 1, "raw_output_object_ids": []},
                {"frame_index": 2, "raw_output_object_ids": []},
            ],
        }
    )

    assert summary == {
        "status": "done",
        "maintained_expected_object_ids": False,
        "first_missing_frame": 1,
        "empty_non_initial_frames": 2,
    }

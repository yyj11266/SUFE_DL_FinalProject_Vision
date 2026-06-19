from __future__ import annotations

import inspect

from scripts.debug_sam31_api import _mask_probe_summary, _patch_model_init_state_filter_kwargs


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def init_state(self, resource_path: str, async_loading_frames: bool = False) -> dict[str, object]:
        self.calls.append(
            {
                "resource_path": resource_path,
                "async_loading_frames": async_loading_frames,
            }
        )
        return {"resource_path": resource_path}


class _FakePredictor:
    def __init__(self) -> None:
        self.model = _FakeModel()


def test_init_state_patch_preserves_signature_and_filters_kwargs() -> None:
    predictor = _FakePredictor()

    status = _patch_model_init_state_filter_kwargs(predictor)
    signature = inspect.signature(predictor.model.init_state)
    state = predictor.model.init_state(
        resource_path="/tmp/video",
        async_loading_frames=True,
        offload_state_to_cpu=True,
    )

    assert status["diagnostic_patch_applied"] is True
    assert "resource_path" in signature.parameters
    assert "offload_state_to_cpu" not in signature.parameters
    assert state == {"resource_path": "/tmp/video"}
    assert predictor.model.calls == [{"resource_path": "/tmp/video", "async_loading_frames": True}]


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


def test_mask_probe_summary_preserves_missing_api_status() -> None:
    summary = _mask_probe_summary(
        {
            "status": "blocked_missing_official_mask_api",
            "api_path": "full_predictor_model.add_new_masks",
            "error": "model does not expose add_new_masks",
        }
    )

    assert summary == {
        "status": "blocked_missing_official_mask_api",
        "maintained_expected_object_ids": False,
        "first_missing_frame": None,
        "empty_non_initial_frames": None,
    }

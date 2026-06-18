"""Motion tracking adapters and gates for video object segmentation."""

from __future__ import annotations

from typing import Any

__all__ = [
    "KalmanBBoxTracker",
    "ObjectPrompt",
    "PreparedFrame",
    "Sam2VideoResult",
    "Sam3Availability",
    "Sam3TrackerBuildResult",
    "Sam3VideoResult",
    "TrackResult",
    "build_object_prompts",
    "build_sam2_video_predictor",
    "build_sam3_tracker",
    "build_sutrack_or_fallback",
    "check_sam3_available",
    "download_sam2_checkpoint",
    "install_or_check_sam2",
    "install_sam3_if_requested",
    "run_sam2_on_video",
    "run_sam3_video_with_mask_prompt",
    "run_sam3_video_with_multi_anchors",
    "track_video_bboxes",
]

_SAM2_EXPORTS = {
    "ObjectPrompt",
    "PreparedFrame",
    "Sam2VideoResult",
    "build_object_prompts",
    "build_sam2_video_predictor",
    "download_sam2_checkpoint",
    "install_or_check_sam2",
    "run_sam2_on_video",
}

_SAM3_EXPORTS = {
    "Sam3Availability",
    "Sam3TrackerBuildResult",
    "Sam3VideoResult",
    "build_sam3_tracker",
    "check_sam3_available",
    "install_sam3_if_requested",
    "run_sam3_video_with_mask_prompt",
    "run_sam3_video_with_multi_anchors",
}

_SUTRACK_EXPORTS = {
    "KalmanBBoxTracker",
    "TrackResult",
    "build_sutrack_or_fallback",
    "track_video_bboxes",
}


def __getattr__(name: str) -> Any:
    """Lazily expose tracker helpers without importing torch at package import time."""

    if name in _SAM2_EXPORTS:
        from src.trackers import sam2_tracker

        return getattr(sam2_tracker, name)
    if name in _SAM3_EXPORTS:
        from src.trackers import sam3_tracker_optional

        return getattr(sam3_tracker_optional, name)
    if name in _SUTRACK_EXPORTS:
        from src.trackers import sutrack_optional

        return getattr(sutrack_optional, name)
    raise AttributeError(f"module 'src.trackers' has no attribute {name!r}")

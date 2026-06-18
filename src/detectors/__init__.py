"""Optional detector adapters for re-detection and distractor mining."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Sam3DetectorBuildResult",
    "Sam3DetectorCandidate",
    "Sam3OptionalDetector",
    "run_sam3_detector_with_text_prompt",
    "run_sam3_detector_with_visual_prompt",
]

_SAM3_DETECTOR_EXPORTS = {
    "Sam3DetectorBuildResult",
    "Sam3DetectorCandidate",
    "Sam3OptionalDetector",
    "run_sam3_detector_with_text_prompt",
    "run_sam3_detector_with_visual_prompt",
}


def __getattr__(name: str) -> Any:
    """Lazily expose optional detector helpers."""

    if name in _SAM3_DETECTOR_EXPORTS:
        from src.detectors import sam3_detector_optional

        return getattr(sam3_detector_optional, name)
    raise AttributeError(f"module 'src.detectors' has no attribute {name!r}")

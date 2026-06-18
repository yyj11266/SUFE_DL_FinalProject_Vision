"""Dataset download, extraction, structure detection, and submission utilities."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DataInfo",
    "FormatSpec",
    "FrameInfo",
    "PromptInfo",
    "VideoInfo",
    "detect_frames",
    "detect_initial_prompts",
    "detect_prompt_format",
    "detect_video_dirs",
    "infer_provisional_format",
    "inspect_dataset",
    "inspect_sample_submission",
    "make_submission",
    "recursively_scan",
    "validate_submission_zip",
]

_INSPECT_EXPORTS = {
    "DataInfo",
    "FrameInfo",
    "PromptInfo",
    "VideoInfo",
    "detect_frames",
    "detect_initial_prompts",
    "detect_prompt_format",
    "detect_video_dirs",
    "inspect_dataset",
    "recursively_scan",
}

_SUBMISSION_EXPORTS = {
    "FormatSpec",
    "infer_provisional_format",
    "inspect_sample_submission",
    "make_submission",
    "validate_submission_zip",
}


def __getattr__(name: str) -> Any:
    """Lazily expose data helpers without interfering with ``python -m`` CLIs."""

    if name in _INSPECT_EXPORTS:
        from src.data import inspect_sufe

        return getattr(inspect_sufe, name)
    if name in _SUBMISSION_EXPORTS:
        from src.data import submission

        return getattr(submission, name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")

"""Prompt parsing, target typing, anchor mining, and prompt fusion utilities."""

from __future__ import annotations

from typing import Any

_TARGET_EXPORTS = {
    "TargetClassifierConfig",
    "TargetProfile",
    "classify_target",
}

_FUSION_EXPORTS = {
    "FramePromptInputs",
    "FusedPrompt",
    "PromptFusionConfig",
    "PromptFusionDebugRecord",
    "fuse_prompt_for_frame",
    "save_prompt_fusion_debug",
}

__all__ = sorted(_TARGET_EXPORTS | _FUSION_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily expose prompt helpers without importing optional modules eagerly."""

    if name in _TARGET_EXPORTS:
        from src.prompts import target_classifier

        return getattr(target_classifier, name)
    if name in _FUSION_EXPORTS:
        from src.prompts import prompt_fusion

        return getattr(prompt_fusion, name)
    raise AttributeError(f"module 'src.prompts' has no attribute {name!r}")

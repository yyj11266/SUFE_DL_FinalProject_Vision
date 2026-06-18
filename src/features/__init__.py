"""Feature extraction adapters for reliability scoring and optional backends."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DinoFeatureBackend",
    "TargetFeaturePool",
    "build_dino_model",
    "extract_augmented_target_pool",
    "extract_crop_feature",
]


def __getattr__(name: str) -> Any:
    """Lazily expose feature helpers."""

    if name in __all__:
        from src.features import dino_features

        return getattr(dino_features, name)
    raise AttributeError(f"module 'src.features' has no attribute {name!r}")

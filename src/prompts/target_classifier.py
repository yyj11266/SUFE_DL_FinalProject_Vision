"""Target type classification for tracking-enhanced prompt fusion."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.vos.reliability import mask_area, mask_to_bbox


@dataclass(slots=True)
class TargetClassifierConfig:
    """Thresholds for target type and semantic-dominated classification."""

    tiny_area_ratio: float = 0.003
    tiny_min_side_px: int = 24
    semantic_top1_threshold: float = 0.70
    semantic_margin_threshold: float = 0.08
    semantic_multi_threshold: float = 0.68
    semantic_multi_count: int = 2


@dataclass(slots=True)
class TargetProfile:
    """Target profile used by prompt fusion and memory policies."""

    target_type: str
    tiny: bool
    semantic_dominated: bool
    area_ratio: float
    scale_ratio: float
    area: int
    bbox: list[int] | None
    frame_size: tuple[int, int]
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable target profile metadata."""

        return asdict(self)


def _as_config(config: TargetClassifierConfig | dict[str, Any] | None) -> TargetClassifierConfig:
    """Normalize target classifier config input."""

    if config is None:
        return TargetClassifierConfig()
    if isinstance(config, TargetClassifierConfig):
        return config
    allowed = TargetClassifierConfig.__dataclass_fields__.keys()
    return TargetClassifierConfig(**{key: value for key, value in config.items() if key in allowed})


def _mask_array(mask: np.ndarray | Image.Image | str | Path) -> np.ndarray:
    """Load a mask-like object to a numpy array."""

    if isinstance(mask, (str, Path)):
        return np.asarray(Image.open(mask))
    if isinstance(mask, Image.Image):
        return np.asarray(mask)
    return np.asarray(mask)


def _frame_size_hw(frame_size: tuple[int, int] | list[int] | np.ndarray) -> tuple[int, int]:
    """Normalize frame size to ``(height, width)``."""

    values = [int(value) for value in list(frame_size)]
    if len(values) != 2:
        raise ValueError(f"frame_size must contain two values, got {frame_size!r}")
    return values[0], values[1]


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from an object or dictionary."""

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _candidate_items(anchor_result: Any = None, candidates_by_frame: Any = None) -> list[Any]:
    """Collect candidate-like records from anchor mining outputs or JSON payloads."""

    items: list[Any] = []
    if candidates_by_frame is not None:
        if isinstance(candidates_by_frame, dict):
            for frame_id, candidates in candidates_by_frame.items():
                for candidate in candidates:
                    if isinstance(candidate, dict) and "frame_id" not in candidate:
                        candidate = {**candidate, "frame_id": frame_id}
                    items.append(candidate)
        else:
            items.extend(list(candidates_by_frame))
    if anchor_result is not None:
        candidates = _read_field(anchor_result, "candidates", None)
        if candidates is None and isinstance(anchor_result, dict):
            candidates = anchor_result.get("candidates")
        if candidates:
            items.extend(list(candidates))
    return items


def _group_candidate_sims(candidates: Iterable[Any]) -> dict[str, list[float]]:
    """Group candidate appearance similarities by frame id."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for item in candidates:
        frame_id = str(_read_field(item, "frame_id", ""))
        sim = _read_field(item, "S_app", _read_field(item, "top1_sim", None))
        if sim is None:
            continue
        try:
            grouped[frame_id].append(float(sim))
        except (TypeError, ValueError):
            continue
    return grouped


def _semantic_dominated(grouped_sims: dict[str, list[float]], config: TargetClassifierConfig) -> tuple[bool, list[str]]:
    """Detect semantic-dominated or distractor-heavy candidate ambiguity."""

    reasons: list[str] = []
    for frame_id, sims in grouped_sims.items():
        sims_sorted = sorted(sims, reverse=True)
        if len(sims_sorted) >= 2:
            top1 = sims_sorted[0]
            top2 = sims_sorted[1]
            if top1 > config.semantic_top1_threshold and top1 - top2 < config.semantic_margin_threshold:
                reasons.append(f"{frame_id}:top1_margin_ambiguous")
        if sum(value > config.semantic_multi_threshold for value in sims_sorted) >= config.semantic_multi_count:
            reasons.append(f"{frame_id}:multiple_high_similarity_candidates")
    return bool(reasons), reasons


def classify_target(
    mask0: np.ndarray | Image.Image | str | Path,
    frame_size: tuple[int, int] | list[int] | np.ndarray,
    anchor_result: Any = None,
    candidates_by_frame: Any = None,
    config: TargetClassifierConfig | dict[str, Any] | None = None,
) -> TargetProfile:
    """Classify the target as tiny, regular, or semantic-dominated."""

    cfg = _as_config(config)
    height, width = _frame_size_hw(frame_size)
    mask = _mask_array(mask0)
    area = mask_area(mask)
    bbox = mask_to_bbox(mask)
    area_ratio = float(area / max(1, height * width))
    if bbox is None:
        scale_ratio = 0.0
        min_side = 0
    else:
        box_width = int(bbox[2] - bbox[0] + 1)
        box_height = int(bbox[3] - bbox[1] + 1)
        min_side = min(box_width, box_height)
        scale_ratio = float(min_side / max(1, max(height, width)))

    tiny = area_ratio < cfg.tiny_area_ratio or min_side < cfg.tiny_min_side_px
    reasons: list[str] = []
    if area_ratio < cfg.tiny_area_ratio:
        reasons.append("tiny_area_ratio")
    if min_side < cfg.tiny_min_side_px:
        reasons.append("tiny_min_side")

    semantic, semantic_reasons = _semantic_dominated(
        _group_candidate_sims(_candidate_items(anchor_result, candidates_by_frame)),
        cfg,
    )
    reasons.extend(semantic_reasons)
    if semantic:
        target_type = "semantic_dominated"
    elif tiny:
        target_type = "tiny"
    else:
        target_type = "regular"
    return TargetProfile(
        target_type=target_type,
        tiny=tiny,
        semantic_dominated=semantic,
        area_ratio=area_ratio,
        scale_ratio=scale_ratio,
        area=area,
        bbox=bbox,
        frame_size=(height, width),
        reasons=reasons,
    )

"""Conservative SAM2/Cutie object-level fusion utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from src.vos.reliability import mask_iou


@dataclass(slots=True)
class ConservativeFusionConfig:
    """Global thresholds for SAM2-anchor/Cutie-candidate fusion."""

    min_cutie_area: int = 16
    min_sam2_iou: float = 0.50
    min_temporal_iou: float = 0.35
    min_area_ratio: float = 0.20
    max_area_ratio: float = 3.50
    allow_cutie_when_sam2_empty: bool = True


@dataclass(slots=True)
class ObjectFusionDecision:
    """One object-level fusion decision."""

    object_id: int
    source: str
    reason: str
    sam2_area: int
    cutie_area: int
    output_area: int
    sam2_cutie_iou: float
    cutie_temporal_iou: float | None
    cutie_area_ratio: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FrameFusionResult:
    """Fused indexed mask and per-object debug decisions."""

    indexed_mask: np.ndarray
    decisions: list[ObjectFusionDecision]
    warnings: list[str] = field(default_factory=list)


def _as_config(config: ConservativeFusionConfig | Mapping[str, Any] | None) -> ConservativeFusionConfig:
    if config is None:
        return ConservativeFusionConfig()
    if isinstance(config, ConservativeFusionConfig):
        return config
    allowed = ConservativeFusionConfig.__dataclass_fields__.keys()
    return ConservativeFusionConfig(**{key: value for key, value in config.items() if key in allowed})


def object_ids_from_indexed(indexed: np.ndarray) -> list[int]:
    """Return positive indexed object ids."""

    return sorted(int(value) for value in np.unique(indexed).tolist() if int(value) > 0)


def validate_known_ids(indexed: np.ndarray, object_ids: Iterable[int]) -> list[int]:
    """Return positive ids in indexed that are not among known object ids."""

    allowed = {int(object_id) for object_id in object_ids}
    return sorted(int(value) for value in np.unique(indexed).tolist() if int(value) > 0 and int(value) not in allowed)


def select_object_mask(
    object_id: int,
    sam2_indexed: np.ndarray,
    cutie_indexed: np.ndarray,
    previous_output: np.ndarray | None = None,
    config: ConservativeFusionConfig | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, ObjectFusionDecision]:
    """Select SAM2 or Cutie mask for one object using global conservative gates."""

    cfg = _as_config(config)
    sam2_mask = np.asarray(sam2_indexed) == int(object_id)
    cutie_mask = np.asarray(cutie_indexed) == int(object_id)
    sam2_area = int(sam2_mask.sum())
    cutie_area = int(cutie_mask.sum())
    warnings: list[str] = []
    sam2_cutie_iou = mask_iou(sam2_mask, cutie_mask)
    temporal_iou: float | None = None
    area_ratio: float | None = None

    if cutie_area < int(cfg.min_cutie_area):
        selected = sam2_mask
        reason = "cutie_empty_or_too_small"
    else:
        if previous_output is not None:
            prev_mask = np.asarray(previous_output) == int(object_id)
            temporal_iou = mask_iou(prev_mask, cutie_mask)
            prev_area = int(prev_mask.sum())
            if prev_area > 0:
                area_ratio = cutie_area / max(1.0, float(prev_area))
        if sam2_area <= 0 and bool(cfg.allow_cutie_when_sam2_empty):
            selected = cutie_mask
            reason = "cutie_replaces_empty_sam2"
        elif sam2_area <= 0:
            selected = sam2_mask
            reason = "sam2_empty_cutie_not_allowed"
        elif sam2_cutie_iou < float(cfg.min_sam2_iou):
            selected = sam2_mask
            reason = "cutie_low_sam2_agreement"
        elif temporal_iou is not None and temporal_iou < float(cfg.min_temporal_iou):
            selected = sam2_mask
            reason = "cutie_low_temporal_iou"
        elif area_ratio is not None and not (float(cfg.min_area_ratio) <= area_ratio <= float(cfg.max_area_ratio)):
            selected = sam2_mask
            reason = "cutie_area_ratio_out_of_range"
        else:
            selected = cutie_mask
            reason = "cutie_passed_conservative_gates"

    source = "cutie" if selected is cutie_mask else "sam2"
    decision = ObjectFusionDecision(
        object_id=int(object_id),
        source=source,
        reason=reason,
        sam2_area=sam2_area,
        cutie_area=cutie_area,
        output_area=int(selected.sum()),
        sam2_cutie_iou=float(sam2_cutie_iou),
        cutie_temporal_iou=temporal_iou,
        cutie_area_ratio=area_ratio,
        warnings=warnings,
    )
    return selected.astype(bool), decision


def compose_object_masks(object_masks: Mapping[int, np.ndarray], object_ids: Iterable[int]) -> np.ndarray:
    """Compose binary object masks into one indexed PNG without creating new ids."""

    ids = [int(object_id) for object_id in object_ids]
    if not ids:
        return np.zeros((0, 0), dtype=np.uint8)
    first_mask = next(iter(object_masks.values()), None)
    if first_mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    output = np.zeros(np.asarray(first_mask).shape[:2], dtype=np.uint8)
    for object_id in ids:
        mask = np.asarray(object_masks.get(object_id, np.zeros_like(first_mask))).astype(bool)
        output[mask & (output == 0)] = np.uint8(min(max(object_id, 0), 255))
    return output


def fuse_frame(
    sam2_indexed: np.ndarray,
    cutie_indexed: np.ndarray,
    object_ids: Iterable[int],
    frame_index: int,
    first_frame_mask: np.ndarray | None = None,
    previous_output: np.ndarray | None = None,
    config: ConservativeFusionConfig | Mapping[str, Any] | None = None,
) -> FrameFusionResult:
    """Fuse one indexed SAM2 frame and one indexed Cutie frame."""

    ids = [int(object_id) for object_id in object_ids]
    warnings: list[str] = []
    if int(frame_index) == 0 and first_frame_mask is not None:
        indexed = np.asarray(first_frame_mask).astype(np.uint8, copy=True)
        decisions = [
            ObjectFusionDecision(
                object_id=object_id,
                source="prompt",
                reason="first_frame_exact",
                sam2_area=int((sam2_indexed == object_id).sum()),
                cutie_area=int((cutie_indexed == object_id).sum()),
                output_area=int((indexed == object_id).sum()),
                sam2_cutie_iou=mask_iou(sam2_indexed == object_id, cutie_indexed == object_id),
                cutie_temporal_iou=None,
                cutie_area_ratio=None,
            )
            for object_id in ids
        ]
        return FrameFusionResult(indexed_mask=indexed, decisions=decisions, warnings=warnings)

    unknown_cutie = validate_known_ids(cutie_indexed, ids)
    if unknown_cutie:
        warnings.append(f"cutie_unknown_ids_cleared:{unknown_cutie[:10]}")
        cutie_indexed = np.asarray(cutie_indexed).copy()
        cutie_indexed[~np.isin(cutie_indexed, [0, *ids])] = 0

    masks: dict[int, np.ndarray] = {}
    decisions: list[ObjectFusionDecision] = []
    for object_id in ids:
        selected, decision = select_object_mask(
            object_id,
            sam2_indexed=sam2_indexed,
            cutie_indexed=cutie_indexed,
            previous_output=previous_output,
            config=config,
        )
        masks[int(object_id)] = selected
        decisions.append(decision)
    return FrameFusionResult(indexed_mask=compose_object_masks(masks, ids), decisions=decisions, warnings=warnings)

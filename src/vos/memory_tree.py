"""Lightweight SAM2Long-like memory tree path search for VOS masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from src.vos.reliability import bbox_iou, mask_area, mask_to_bbox


BBox = list[float] | tuple[float, float, float, float] | np.ndarray | None
MaskLike = np.ndarray | Image.Image | str | Path | None
LogitLike = np.ndarray | str | Path | None


@dataclass(slots=True)
class MemoryTreeConfig:
    """Configuration for lightweight constrained path search."""

    lambda_decay: float = 0.98
    eta: float = 0.25
    top_k_paths: int = 3
    eps: float = 1e-6
    min_area: int = 16
    stable_threshold: float = 0.65
    lost_threshold: float = 0.40
    empty_objectness_threshold: float = 0.30
    empty_tracker_conf_threshold: float = 0.35
    empty_high_score: float = 0.45
    empty_low_score: float = 0.10
    allow_empty_branches: bool = True
    postprocess_kernel_size: int = 3
    postprocess_min_component_area: int = 16
    write_outputs: bool = True


@dataclass(slots=True)
class MaskCandidate:
    """Candidate mask hypothesis for one frame and object."""

    frame_id: str | int
    frame_index: int
    object_id: str | int
    source: str
    mask: MaskLike
    bbox: list[float] | None = None
    logit: LogitLike = None
    reliability: float | Any | None = None
    objectness: float | None = None
    tracker_confidence: float | None = None
    state: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary of this candidate."""

        mask = _load_mask(self.mask)
        bbox = self.bbox if self.bbox is not None else mask_to_bbox(mask)
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "source": self.source,
            "mask": _mask_ref(self.mask),
            "bbox": _json_box(bbox),
            "logit": _logit_ref(self.logit),
            "reliability": _reliability_value(self.reliability),
            "objectness": self.objectness,
            "tracker_confidence": self.tracker_confidence,
            "state": self.state,
            "area": mask_area(mask),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CandidatePath:
    """One retained path through frame-level mask candidates."""

    path_id: str
    object_id: str | int
    masks: dict[str | int, MaskLike]
    bboxes: dict[str | int, list[float] | None]
    logits: dict[str | int, LogitLike] | None = None
    reliability_scores: dict[str | int, float] = field(default_factory=dict)
    cumulative_score: float = 0.0
    last_good_frame: int = -1
    state: str = "stable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe summary without embedding mask arrays."""

        logits = self.logits or {}
        return {
            "path_id": self.path_id,
            "object_id": self.object_id,
            "masks": {str(key): _mask_ref(value) for key, value in self.masks.items()},
            "bboxes": {str(key): _json_box(value) for key, value in self.bboxes.items()},
            "logits": {str(key): _logit_ref(value) for key, value in logits.items()},
            "reliability_scores": {str(key): float(value) for key, value in self.reliability_scores.items()},
            "cumulative_score": float(self.cumulative_score),
            "last_good_frame": int(self.last_good_frame),
            "state": self.state,
            "metadata": _json_safe_metadata(self.metadata),
        }


@dataclass(slots=True)
class MemoryTreeResult:
    """Output of a memory tree constrained path search."""

    best_path: CandidatePath
    active_paths_by_frame: dict[str | int, list[CandidatePath]]
    candidate_scores_by_frame: dict[str | int, list[dict[str, Any]]]
    warnings: list[str] = field(default_factory=list)
    best_mask_dir: str | None = None
    debug_json_path: str | None = None
    path_scores_csv_path: str | None = None
    path_scores_png_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result summary."""

        return {
            "best_path": self.best_path.to_dict(),
            "active_paths_by_frame": {
                str(frame_id): [path.to_dict() for path in paths]
                for frame_id, paths in self.active_paths_by_frame.items()
            },
            "candidate_scores_by_frame": {
                str(frame_id): rows for frame_id, rows in self.candidate_scores_by_frame.items()
            },
            "warnings": self.warnings,
            "best_mask_dir": self.best_mask_dir,
            "debug_json_path": self.debug_json_path,
            "path_scores_csv_path": self.path_scores_csv_path,
            "path_scores_png_path": self.path_scores_png_path,
        }


def _as_config(config: MemoryTreeConfig | dict[str, Any] | None) -> MemoryTreeConfig:
    """Normalize config input to ``MemoryTreeConfig``."""

    if config is None:
        return MemoryTreeConfig()
    if isinstance(config, MemoryTreeConfig):
        return config
    allowed = MemoryTreeConfig.__dataclass_fields__.keys()
    return MemoryTreeConfig(**{key: value for key, value in config.items() if key in allowed})


def _natural_key(value: Any) -> list[int | str]:
    """Sort frame ids with embedded numbers in numeric order."""

    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_mask(mask: MaskLike, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load a mask-like object as a boolean numpy array."""

    if mask is None:
        if size is None:
            return np.zeros((0, 0), dtype=bool)
        return np.zeros((size[1], size[0]), dtype=bool)
    if isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        foreground = np.any(array[..., :3] > 0, axis=-1)
    else:
        foreground = array > 0
    if size is not None and foreground.shape != (size[1], size[0]):
        foreground = np.asarray(
            Image.fromarray(foreground.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
        ) > 0
    return foreground.astype(bool)


def _load_logit(logit: LogitLike) -> np.ndarray | None:
    """Load a logit-like object when it is needed for output propagation."""

    if logit is None:
        return None
    if isinstance(logit, np.ndarray):
        return logit
    path = Path(logit)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            first_key = sorted(payload.files)[0]
            return np.asarray(payload[first_key])
    return None


def _normalize_box(box: BBox) -> list[float] | None:
    """Normalize bbox to inclusive float coordinates."""

    if box is None:
        return None
    try:
        values = np.asarray(box, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if values.size != 4 or not np.all(np.isfinite(values)):
        return None
    x0, y0, x1, y1 = [float(value) for value in values.tolist()]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _json_box(box: BBox) -> list[float] | None:
    """Return a JSON-safe bbox list."""

    normalized = _normalize_box(box)
    return None if normalized is None else [float(value) for value in normalized]


def _bbox_area(box: BBox) -> float:
    """Return inclusive bbox area."""

    normalized = _normalize_box(box)
    if normalized is None:
        return 0.0
    return max(0.0, normalized[2] - normalized[0] + 1.0) * max(0.0, normalized[3] - normalized[1] + 1.0)


def _mask_ref(mask: MaskLike) -> dict[str, Any] | str | None:
    """Summarize a mask reference without serializing array payloads."""

    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return str(mask)
    if isinstance(mask, Image.Image):
        return {"type": "PIL.Image", "size": list(mask.size)}
    array = np.asarray(mask)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _logit_ref(logit: LogitLike) -> dict[str, Any] | str | None:
    """Summarize a logit reference without serializing array payloads."""

    if logit is None:
        return None
    if isinstance(logit, (str, Path)):
        return str(logit)
    array = np.asarray(logit)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _json_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return metadata with heavy mask/logit entries summarized."""

    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"seed_mask"}:
            safe[key] = _mask_ref(value)
        elif key in {"seed_bbox"}:
            safe[key] = _json_box(value)
        elif key in {"frame_order"}:
            safe[key] = [str(item) for item in value]
        elif isinstance(value, np.ndarray):
            safe[key] = {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, dict):
            safe[key] = {str(k): _json_safe_value(v) for k, v in value.items()}
        else:
            safe[key] = _json_safe_value(value)
    return safe


def _json_safe_value(value: Any) -> Any:
    """Convert common numpy/path values to JSON-safe primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _read_field(item: Any, *names: str, default: Any = None) -> Any:
    """Read a field from a dataclass-like object or dictionary."""

    if item is None:
        return default
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _reliability_value(value: float | Any | None) -> float | None:
    """Extract a reliability float from a scalar or ReliabilityResult-like object."""

    if value is None:
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    reliability = _read_field(value, "reliability", default=None)
    if reliability is None:
        return None
    try:
        return float(reliability)
    except (TypeError, ValueError):
        return None


def _state_from_score(reliability: float, area: int, config: MemoryTreeConfig, source: str) -> str:
    """Classify candidate state from reliability, area, and source."""

    if source == "empty" or area < config.min_area or reliability < config.lost_threshold:
        return "lost"
    if reliability >= config.stable_threshold:
        return "stable"
    return "ambiguous"


def _postprocess_mask(mask: np.ndarray, config: MemoryTreeConfig) -> np.ndarray:
    """Apply lightweight morphological cleanup to a candidate mask."""

    if mask.size == 0:
        return mask.astype(bool)
    kernel_size = max(1, int(config.postprocess_kernel_size))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    output = np.zeros_like(cleaned, dtype=np.uint8)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(config.postprocess_min_component_area):
            output[labels == label] = 1
    return output.astype(bool)


def build_frame_candidates(
    frame_id: str | int,
    frame_index: int,
    object_id: str | int,
    primary_mask: MaskLike,
    alternative_masks: Sequence[MaskLike] | MaskLike | None = None,
    prompt_fusion_mask: MaskLike = None,
    primary_reliability: float | Any | None = None,
    alternative_reliabilities: Sequence[float | Any | None] | float | Any | None = None,
    prompt_fusion_reliability: float | Any | None = None,
    primary_logit: LogitLike = None,
    alternative_logits: Sequence[LogitLike] | LogitLike | None = None,
    prompt_fusion_logit: LogitLike = None,
    objectness: float | None = None,
    tracker_confidence: float | None = None,
    state: str | None = None,
    config: MemoryTreeConfig | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[MaskCandidate]:
    """Build standard per-frame candidates from SAM and prompt-fusion masks."""

    cfg = _as_config(config)
    frame_metadata = dict(metadata or {})
    candidates: list[MaskCandidate] = []
    primary = _load_mask(primary_mask)
    primary_bbox = _json_box(mask_to_bbox(primary))
    candidates.append(
        MaskCandidate(
            frame_id=frame_id,
            frame_index=frame_index,
            object_id=object_id,
            source="primary_sam",
            mask=primary_mask,
            bbox=primary_bbox,
            logit=primary_logit,
            reliability=primary_reliability,
            objectness=objectness,
            tracker_confidence=tracker_confidence,
            state=state,
            metadata=frame_metadata,
        )
    )
    postprocessed = _postprocess_mask(primary, cfg)
    candidates.append(
        MaskCandidate(
            frame_id=frame_id,
            frame_index=frame_index,
            object_id=object_id,
            source="postprocessed_primary",
            mask=postprocessed,
            bbox=_json_box(mask_to_bbox(postprocessed)),
            logit=primary_logit,
            reliability=primary_reliability,
            objectness=objectness,
            tracker_confidence=tracker_confidence,
            state=state,
            metadata={**frame_metadata, "derived_from": "primary_sam"},
        )
    )
    alt_masks = _as_sequence(alternative_masks)
    alt_rels = _as_sequence(alternative_reliabilities)
    alt_logits = _as_sequence(alternative_logits)
    for index, alt_mask in enumerate(alt_masks):
        alt_loaded = _load_mask(alt_mask)
        candidates.append(
            MaskCandidate(
                frame_id=frame_id,
                frame_index=frame_index,
                object_id=object_id,
                source="alternative_sam",
                mask=alt_mask,
                bbox=_json_box(mask_to_bbox(alt_loaded)),
                logit=alt_logits[index] if index < len(alt_logits) else None,
                reliability=alt_rels[index] if index < len(alt_rels) else primary_reliability,
                objectness=objectness,
                tracker_confidence=tracker_confidence,
                state=state,
                metadata={**frame_metadata, "alternative_index": index},
            )
        )
    if prompt_fusion_mask is not None:
        prompt_loaded = _load_mask(prompt_fusion_mask)
        candidates.append(
            MaskCandidate(
                frame_id=frame_id,
                frame_index=frame_index,
                object_id=object_id,
                source="prompt_fusion_corrected",
                mask=prompt_fusion_mask,
                bbox=_json_box(mask_to_bbox(prompt_loaded)),
                logit=prompt_fusion_logit,
                reliability=prompt_fusion_reliability if prompt_fusion_reliability is not None else primary_reliability,
                objectness=objectness,
                tracker_confidence=tracker_confidence,
                state=state,
                metadata=frame_metadata,
            )
        )
    candidates.append(
        MaskCandidate(
            frame_id=frame_id,
            frame_index=frame_index,
            object_id=object_id,
            source="empty",
            mask=None,
            bbox=None,
            logit=None,
            reliability=None,
            objectness=objectness,
            tracker_confidence=tracker_confidence,
            state=state,
            metadata={**frame_metadata, "absent_hypothesis": True},
        )
    )
    return candidates


def _as_sequence(value: Any) -> list[Any]:
    """Return a value as a list while preserving None as empty."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _candidate_from_any(item: MaskCandidate | dict[str, Any], frame_id: str | int | None = None, frame_index: int | None = None) -> MaskCandidate:
    """Normalize candidate-like input to ``MaskCandidate``."""

    if isinstance(item, MaskCandidate):
        return item
    item_frame_id = item.get("frame_id", frame_id)
    if item_frame_id is None:
        raise ValueError("candidate is missing frame_id")
    item_frame_index = item.get("frame_index", frame_index)
    if item_frame_index is None:
        item_frame_index = _frame_index_from_id(item_frame_id)
    mask = item.get("mask", item.get("mask_path"))
    logit = item.get("logit", item.get("logit_path"))
    reliability = item.get("reliability", item.get("R_t"))
    return MaskCandidate(
        frame_id=item_frame_id,
        frame_index=int(item_frame_index),
        object_id=item.get("object_id", 1),
        source=str(item.get("source", "primary_sam")),
        mask=mask,
        bbox=_json_box(item.get("bbox")),
        logit=logit,
        reliability=reliability,
        objectness=_optional_float(item.get("objectness")),
        tracker_confidence=_optional_float(item.get("tracker_confidence", item.get("tracker_conf"))),
        state=item.get("state"),
        metadata=dict(item.get("metadata", {})),
    )


def _frame_index_from_id(frame_id: str | int) -> int:
    """Infer a numeric frame index from common frame ids."""

    if isinstance(frame_id, int):
        return frame_id
    digits = "".join(char for char in str(frame_id) if char.isdigit())
    return int(digits) if digits else 0


def _optional_float(value: Any) -> float | None:
    """Convert a value to float or None."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_candidates_by_frame(
    candidates_by_frame: Mapping[Any, Iterable[MaskCandidate | dict[str, Any]]] | Iterable[MaskCandidate | dict[str, Any]],
) -> list[tuple[str | int, list[MaskCandidate]]]:
    """Normalize candidate input into sorted frame groups."""

    grouped: dict[str | int, list[MaskCandidate]] = {}
    if isinstance(candidates_by_frame, Mapping):
        for offset, (frame_id, items) in enumerate(candidates_by_frame.items()):
            frame_candidates = [_candidate_from_any(item, frame_id=frame_id, frame_index=offset) for item in items]
            grouped[frame_id] = frame_candidates
    else:
        for item in candidates_by_frame:
            candidate = _candidate_from_any(item)
            grouped.setdefault(candidate.frame_id, []).append(candidate)
    return sorted(grouped.items(), key=lambda pair: (min((candidate.frame_index for candidate in pair[1]), default=0), _natural_key(pair[0])))


def _candidate_matches_object(candidate: MaskCandidate, object_id: str | int) -> bool:
    """Return whether a candidate belongs to the requested object."""

    return str(candidate.object_id) == str(object_id)


def _empty_allowed(candidate: MaskCandidate, area: int, reliability: float, config: MemoryTreeConfig) -> bool:
    """Return whether an empty branch should participate in path expansion."""

    if candidate.source != "empty":
        return True
    if not config.allow_empty_branches:
        return False
    objectness = 0.0 if candidate.objectness is None else float(candidate.objectness)
    tracker_confidence = 0.0 if candidate.tracker_confidence is None else float(candidate.tracker_confidence)
    return bool(
        candidate.state == "lost"
        and objectness <= config.empty_objectness_threshold
        and area < config.min_area
        and tracker_confidence <= config.empty_tracker_conf_threshold
    )


def _candidate_reliability(candidate: MaskCandidate, area: int, config: MemoryTreeConfig) -> float:
    """Return candidate reliability with empty-branch policy applied."""

    if candidate.source == "empty":
        objectness = 0.0 if candidate.objectness is None else float(candidate.objectness)
        tracker_confidence = 0.0 if candidate.tracker_confidence is None else float(candidate.tracker_confidence)
        lost_confidence_high = bool(
            candidate.state == "lost"
            and objectness <= config.empty_objectness_threshold
            and area < config.min_area
            and tracker_confidence <= config.empty_tracker_conf_threshold
        )
        return config.empty_high_score if lost_confidence_high else config.empty_low_score
    reliability = _reliability_value(candidate.reliability)
    return float(reliability if reliability is not None else 0.0)


def _previous_mask(path: CandidatePath) -> MaskLike:
    """Return latest mask in a path, falling back to an initial seed mask."""

    frame_order = path.metadata.get("frame_order", [])
    if frame_order:
        return path.masks[frame_order[-1]]
    return path.metadata.get("seed_mask")


def _previous_bbox(path: CandidatePath) -> list[float] | None:
    """Return latest bbox in a path, falling back to an initial seed bbox."""

    frame_order = path.metadata.get("frame_order", [])
    if frame_order:
        return path.bboxes.get(frame_order[-1])
    return _json_box(path.metadata.get("seed_bbox"))


def _predict_motion_bbox(
    path: CandidatePath,
    frame_id: str | int,
    frame_index: int,
    motion_bboxes: Mapping[Any, BBox] | None,
) -> list[float] | None:
    """Predict current bbox from external motion bboxes or path history."""

    if motion_bboxes is not None:
        if frame_id in motion_bboxes:
            return _json_box(motion_bboxes[frame_id])
        if str(frame_id) in motion_bboxes:
            return _json_box(motion_bboxes[str(frame_id)])
        if frame_index in motion_bboxes:
            return _json_box(motion_bboxes[frame_index])
    frame_order = list(path.metadata.get("frame_order", []))
    frame_indices = path.metadata.get("frame_indices", {})
    history: list[tuple[int, list[float]]] = []
    seed_bbox = _json_box(path.metadata.get("seed_bbox"))
    if seed_bbox is not None:
        history.append((-1, seed_bbox))
    for item in frame_order:
        box = _json_box(path.bboxes.get(item))
        if box is None:
            continue
        history.append((int(frame_indices.get(str(item), frame_indices.get(item, len(history)))), box))
    if not history:
        return None
    if len(history) == 1:
        return history[-1][1]
    prev_index, prev_box = history[-1]
    prev2_index, prev2_box = history[-2]
    dt_prev = max(1, prev_index - prev2_index)
    dt_current = max(1, frame_index - prev_index)
    velocity = (np.asarray(prev_box, dtype=np.float32) - np.asarray(prev2_box, dtype=np.float32)) / float(dt_prev)
    prediction = np.asarray(prev_box, dtype=np.float32) + velocity * float(dt_current)
    return [float(value) for value in prediction.tolist()]


def _compute_drift(
    path: CandidatePath,
    candidate: MaskCandidate,
    mask: np.ndarray,
    bbox: list[float] | None,
    motion_bbox: list[float] | None,
    config: MemoryTreeConfig,
) -> tuple[float, float, float]:
    """Compute drift score and its area/motion components."""

    if candidate.source == "empty":
        return 0.0, 0.0, 0.0
    previous = _previous_mask(path)
    if previous is None:
        return 0.0, 0.0, 0.0
    prev_mask = _load_mask(previous)
    if prev_mask.size == 0:
        return 0.0, 0.0, 0.0
    current_area = int(mask.sum())
    previous_area = int(prev_mask.sum())
    area_jump = abs(math.log((current_area + config.eps) / (previous_area + config.eps)))
    if bbox is None or motion_bbox is None:
        motion_penalty = 0.0
    else:
        motion_penalty = 1.0 - bbox_iou(bbox, motion_bbox)
    return float(area_jump + motion_penalty), float(area_jump), float(motion_penalty)


def _clone_extend_path(
    parent: CandidatePath,
    candidate: MaskCandidate,
    mask: MaskLike,
    bbox: list[float] | None,
    reliability: float,
    cumulative_score: float,
    state: str,
    drift: float,
    path_id: str,
) -> CandidatePath:
    """Return a shallow-copy path extended by one candidate."""

    masks = dict(parent.masks)
    bboxes = dict(parent.bboxes)
    logits = dict(parent.logits or {})
    reliability_scores = dict(parent.reliability_scores)
    metadata = {**parent.metadata}
    frame_order = list(metadata.get("frame_order", []))
    frame_indices = dict(metadata.get("frame_indices", {}))
    sources_by_frame = dict(metadata.get("sources_by_frame", {}))
    drift_by_frame = dict(metadata.get("drift_by_frame", {}))
    candidate_metadata_by_frame = dict(metadata.get("candidate_metadata_by_frame", {}))
    masks[candidate.frame_id] = mask
    bboxes[candidate.frame_id] = bbox
    if candidate.logit is not None:
        logits[candidate.frame_id] = candidate.logit
    reliability_scores[candidate.frame_id] = float(reliability)
    frame_order.append(candidate.frame_id)
    frame_indices[str(candidate.frame_id)] = int(candidate.frame_index)
    sources_by_frame[str(candidate.frame_id)] = candidate.source
    drift_by_frame[str(candidate.frame_id)] = float(drift)
    candidate_metadata_by_frame[str(candidate.frame_id)] = _json_safe_metadata(candidate.metadata)
    metadata.update(
        {
            "frame_order": frame_order,
            "frame_indices": frame_indices,
            "sources_by_frame": sources_by_frame,
            "drift_by_frame": drift_by_frame,
            "candidate_metadata_by_frame": candidate_metadata_by_frame,
        }
    )
    last_good_frame = parent.last_good_frame
    if state == "stable" and candidate.source != "empty" and mask_area(mask) >= 1:
        last_good_frame = int(candidate.frame_index)
    return CandidatePath(
        path_id=path_id,
        object_id=parent.object_id,
        masks=masks,
        bboxes=bboxes,
        logits=logits,
        reliability_scores=reliability_scores,
        cumulative_score=float(cumulative_score),
        last_good_frame=last_good_frame,
        state=state,
        metadata=metadata,
    )


def run_memory_tree_search(
    candidates_by_frame: Mapping[Any, Iterable[MaskCandidate | dict[str, Any]]] | Iterable[MaskCandidate | dict[str, Any]],
    object_id: str | int,
    output_dir: str | Path,
    initial_mask: MaskLike = None,
    motion_bboxes: Mapping[Any, BBox] | None = None,
    config: MemoryTreeConfig | dict[str, Any] | None = None,
) -> MemoryTreeResult:
    """Run lightweight constrained path search over mask candidates."""

    cfg = _as_config(config)
    output_path = Path(output_dir)
    warnings: list[str] = []
    frame_groups = _normalize_candidates_by_frame(candidates_by_frame)
    if not frame_groups:
        raise ValueError("candidates_by_frame is empty")
    seed_mask = _load_mask(initial_mask) if initial_mask is not None else None
    seed_bbox = _json_box(mask_to_bbox(seed_mask)) if seed_mask is not None else None
    root = CandidatePath(
        path_id=f"{object_id}_root",
        object_id=object_id,
        masks={},
        bboxes={},
        logits={},
        reliability_scores={},
        cumulative_score=0.0,
        last_good_frame=-1,
        state="stable",
        metadata={"frame_order": [], "frame_indices": {}, "seed_mask": initial_mask, "seed_bbox": seed_bbox},
    )
    active_paths = [root]
    active_paths_by_frame: dict[str | int, list[CandidatePath]] = {}
    candidate_scores_by_frame: dict[str | int, list[dict[str, Any]]] = {}
    path_counter = 0
    for frame_id, raw_candidates in frame_groups:
        usable_candidates = [candidate for candidate in raw_candidates if _candidate_matches_object(candidate, object_id)]
        if not usable_candidates:
            warnings.append(f"{frame_id}: no candidates for object_id={object_id}")
            active_paths_by_frame[frame_id] = active_paths
            candidate_scores_by_frame[frame_id] = []
            continue
        expanded: list[CandidatePath] = []
        frame_rows: list[dict[str, Any]] = []
        mask_cache: dict[int, np.ndarray] = {}
        for parent in active_paths:
            for candidate in usable_candidates:
                candidate_key = id(candidate)
                if candidate.source == "empty":
                    mask = _load_mask(candidate.mask)
                elif candidate_key in mask_cache:
                    mask = mask_cache[candidate_key]
                else:
                    mask = _load_mask(candidate.mask)
                    mask_cache[candidate_key] = mask
                if candidate.source == "empty" and mask.size == 0:
                    previous = _previous_mask(parent)
                    if previous is not None:
                        prev_mask = _load_mask(previous)
                        mask = np.zeros_like(prev_mask, dtype=bool)
                    elif seed_mask is not None:
                        mask = np.zeros_like(seed_mask, dtype=bool)
                bbox = _json_box(candidate.bbox if candidate.bbox is not None else mask_to_bbox(mask))
                area = int(mask.sum())
                reliability = _candidate_reliability(candidate, area, cfg)
                if candidate.source == "empty" and not _empty_allowed(candidate, area, reliability, cfg):
                    frame_rows.append(
                        _score_row(
                            candidate=candidate,
                            parent_path_id=parent.path_id,
                            path_id="skipped_empty",
                            bbox=bbox,
                            motion_bbox=None,
                            reliability=reliability,
                            drift=0.0,
                            area_jump=0.0,
                            motion_penalty=0.0,
                            cumulative_score=-1.0e12,
                            area=area,
                            pruned=True,
                            note="empty_branch_not_allowed",
                        )
                    )
                    continue
                motion_bbox = _predict_motion_bbox(parent, candidate.frame_id, candidate.frame_index, motion_bboxes)
                drift, area_jump, motion_penalty = _compute_drift(parent, candidate, mask, bbox, motion_bbox, cfg)
                cumulative = cfg.lambda_decay * parent.cumulative_score + reliability - cfg.eta * drift
                state = candidate.state or _state_from_score(reliability, area, cfg, candidate.source)
                path_counter += 1
                candidate_path_id = f"{object_id}_p{path_counter:06d}"
                child = _clone_extend_path(
                    parent=parent,
                    candidate=candidate,
                    mask=mask,
                    bbox=bbox,
                    reliability=reliability,
                    cumulative_score=cumulative,
                    state=state,
                    drift=drift,
                    path_id=candidate_path_id,
                )
                expanded.append(child)
                frame_rows.append(
                    _score_row(
                        candidate=candidate,
                        parent_path_id=parent.path_id,
                        path_id=candidate_path_id,
                        bbox=bbox,
                        motion_bbox=motion_bbox,
                        reliability=reliability,
                        drift=drift,
                        area_jump=area_jump,
                        motion_penalty=motion_penalty,
                        cumulative_score=cumulative,
                        area=area,
                        pruned=False,
                        note="",
                    )
                )
        if not expanded:
            warnings.append(f"{frame_id}: all candidates were pruned; preserving previous active paths")
            active_paths_by_frame[frame_id] = active_paths
            candidate_scores_by_frame[frame_id] = frame_rows
            continue
        expanded_sorted = sorted(expanded, key=lambda path: path.cumulative_score, reverse=True)
        active_paths = expanded_sorted[: max(1, int(cfg.top_k_paths))]
        retained_ids = {path.path_id for path in active_paths}
        for row in frame_rows:
            if row["path_id"] not in retained_ids and not row["pruned"]:
                row["pruned"] = True
                row["note"] = "topk_pruned"
        active_paths_by_frame[frame_id] = list(active_paths)
        candidate_scores_by_frame[frame_id] = frame_rows
    best_path = max(active_paths, key=lambda path: path.cumulative_score)
    result = MemoryTreeResult(
        best_path=best_path,
        active_paths_by_frame=active_paths_by_frame,
        candidate_scores_by_frame=candidate_scores_by_frame,
        warnings=warnings,
    )
    if cfg.write_outputs:
        _write_outputs(result, output_path, cfg)
    return result


def _score_row(
    candidate: MaskCandidate,
    parent_path_id: str,
    path_id: str,
    bbox: list[float] | None,
    motion_bbox: list[float] | None,
    reliability: float,
    drift: float,
    area_jump: float,
    motion_penalty: float,
    cumulative_score: float,
    area: int,
    pruned: bool,
    note: str,
) -> dict[str, Any]:
    """Build one path-score debug row."""

    return {
        "frame_id": candidate.frame_id,
        "frame_index": candidate.frame_index,
        "object_id": candidate.object_id,
        "parent_path_id": parent_path_id,
        "path_id": path_id,
        "source": candidate.source,
        "reliability": float(reliability),
        "drift": float(drift),
        "area_jump": float(area_jump),
        "motion_penalty": float(motion_penalty),
        "cumulative_score": float(cumulative_score),
        "area": int(area),
        "bbox": _json_box(bbox),
        "motion_bbox": _json_box(motion_bbox),
        "objectness": candidate.objectness,
        "tracker_confidence": candidate.tracker_confidence,
        "state": candidate.state,
        "pruned": bool(pruned),
        "note": note,
    }


def _write_outputs(result: MemoryTreeResult, output_dir: Path, config: MemoryTreeConfig) -> None:
    """Write best masks and memory tree debug artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    best_mask_dir = output_dir / "best_masks"
    best_mask_dir.mkdir(parents=True, exist_ok=True)
    frame_order = list(result.best_path.metadata.get("frame_order", result.best_path.masks.keys()))
    for frame_id in frame_order:
        mask = _load_mask(result.best_path.masks[frame_id])
        Image.fromarray((mask.astype(np.uint8) * 255)).save(best_mask_dir / f"{frame_id}.png")
    debug_path = output_dir / "memory_tree_debug.json"
    csv_path = output_dir / "path_scores.csv"
    png_path = output_dir / "path_scores.png"
    result.best_mask_dir = str(best_mask_dir)
    result.debug_json_path = str(debug_path)
    result.path_scores_csv_path = str(csv_path)
    result.path_scores_png_path = str(png_path)
    debug_path.write_text(
        json.dumps({"config": asdict(config), **result.to_dict()}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    _write_scores_csv(result, csv_path)
    _write_scores_png(result, png_path)


def _write_scores_csv(result: MemoryTreeResult, output_path: Path) -> None:
    """Write all candidate extension scores as CSV."""

    rows = [row for frame_rows in result.candidate_scores_by_frame.values() for row in frame_rows]
    fieldnames = [
        "frame_id",
        "frame_index",
        "object_id",
        "parent_path_id",
        "path_id",
        "source",
        "reliability",
        "drift",
        "area_jump",
        "motion_penalty",
        "cumulative_score",
        "area",
        "bbox",
        "motion_bbox",
        "objectness",
        "tracker_confidence",
        "state",
        "pruned",
        "note",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["bbox"] = json.dumps(csv_row["bbox"])
            csv_row["motion_bbox"] = json.dumps(csv_row["motion_bbox"])
            writer.writerow(csv_row)


def _write_scores_png(result: MemoryTreeResult, output_path: Path) -> None:
    """Render retained path scores over time."""

    try:
        cache_dir = output_path.parent / ".matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        result.warnings.append(f"path_scores.png skipped: matplotlib unavailable ({type(exc).__name__}: {exc})")
        return
    frame_ids = list(result.active_paths_by_frame.keys())
    fig, ax = plt.subplots(figsize=(max(6, len(frame_ids) * 0.8), 4))
    for rank in range(max(len(paths) for paths in result.active_paths_by_frame.values())):
        xs: list[int] = []
        ys: list[float] = []
        labels: list[str] = []
        for x, frame_id in enumerate(frame_ids):
            paths = result.active_paths_by_frame[frame_id]
            if rank >= len(paths):
                continue
            xs.append(x)
            ys.append(float(paths[rank].cumulative_score))
            labels.append(paths[rank].path_id)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"rank{rank + 1}")
            for x, y, label in zip(xs, ys, labels):
                ax.annotate(label.split("_")[-1], (x, y), fontsize=6)
    ax.set_xticks(list(range(len(frame_ids))))
    ax.set_xticklabels([str(frame_id) for frame_id in frame_ids], rotation=45, ha="right")
    ax.set_xlabel("frame")
    ax.set_ylabel("cumulative score")
    ax.set_title("Memory tree retained path scores")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _load_candidates_json(path: str | Path) -> dict[Any, list[dict[str, Any]]]:
    """Load CLI candidates JSON into a frame mapping."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "frames" in payload:
        output: dict[Any, list[dict[str, Any]]] = {}
        for offset, frame in enumerate(payload["frames"]):
            frame_id = frame.get("frame_id", offset)
            candidates = frame.get("candidates", [])
            output[frame_id] = [{**candidate, "frame_id": candidate.get("frame_id", frame_id)} for candidate in candidates]
        return output
    if isinstance(payload, dict):
        return {
            frame_id: candidates if isinstance(candidates, list) else candidates.get("candidates", [])
            for frame_id, candidates in payload.items()
        }
    if isinstance(payload, list):
        output = {}
        for candidate in payload:
            output.setdefault(candidate.get("frame_id", 0), []).append(candidate)
        return output
    raise ValueError(f"Unsupported candidates JSON format: {type(payload).__name__}")


def _load_motion_bboxes_json(path: str | Path | None) -> dict[Any, BBox] | None:
    """Load optional motion bbox JSON mapping."""

    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "bboxes" in payload:
        payload = payload["bboxes"]
    if not isinstance(payload, dict):
        raise ValueError("motion bboxes JSON must be a mapping or contain a 'bboxes' mapping")
    return payload


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for memory tree debug runs."""

    parser = argparse.ArgumentParser(description="Run lightweight SAM2Long-like memory tree path search.")
    parser.add_argument("--candidates-json", required=True, help="JSON containing frame candidate mask paths.")
    parser.add_argument("--output-dir", required=True, help="Directory for best_masks and debug artifacts.")
    parser.add_argument("--object-id", required=True, help="Object id to search.")
    parser.add_argument("--initial-mask", default=None, help="Optional initial object mask path.")
    parser.add_argument("--motion-bboxes-json", default=None, help="Optional frame_id -> bbox JSON mapping.")
    parser.add_argument("--lambda-decay", type=float, default=0.98)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--top-k-paths", type=int, default=3)
    parser.add_argument("--min-area", type=int, default=16)
    parser.add_argument("--no-write-outputs", action="store_true", help="Run search without writing artifacts.")
    return parser.parse_args()


def main() -> None:
    """Run the memory tree CLI."""

    args = _parse_args()
    config = MemoryTreeConfig(
        lambda_decay=args.lambda_decay,
        eta=args.eta,
        top_k_paths=args.top_k_paths,
        min_area=args.min_area,
        write_outputs=not args.no_write_outputs,
    )
    result = run_memory_tree_search(
        candidates_by_frame=_load_candidates_json(args.candidates_json),
        object_id=args.object_id,
        output_dir=args.output_dir,
        initial_mask=args.initial_mask,
        motion_bboxes=_load_motion_bboxes_json(args.motion_bboxes_json),
        config=config,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""Tracking-enhanced prompt fusion for corrective VOS prompts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.vos.reliability import bbox_iou


BBox = list[float] | tuple[float, float, float, float] | np.ndarray | None


@dataclass(slots=True)
class PromptFusionConfig:
    """Thresholds for tracking-enhanced prompt fusion decisions."""

    tau_iou: float = 0.35
    tau_conf: float = 0.55
    tau_R: float = 0.65
    regular_reliability_threshold: float = 0.45
    regular_area_jump_threshold: float = 1.0
    tiny_area_drop_ratio: float = 0.30
    max_negative_points: int = 4
    distractor_similarity_threshold: float = 0.68
    negative_exclude_iou: float = 0.50


@dataclass(slots=True)
class FramePromptInputs:
    """Inputs needed to fuse one frame's corrective prompt."""

    frame_id: str | int
    object_id: str | int
    B_sam: BBox
    B_aux: BBox = None
    c_aux: float = 0.0
    R_t: float = 1.0
    sam_area: float | None = None
    expected_area: float | None = None
    area_jump: float = 0.0
    previous_good_bbox: BBox = None
    candidates: Iterable[Any] | None = None
    distractor_candidates: Iterable[Any] | None = None
    aux_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptFusionDebugRecord:
    """JSON debug record for one prompt fusion decision."""

    frame_id: str | int
    object_id: str | int
    target_type: str
    B_sam: list[float] | None
    B_aux: list[float] | None
    IoU_aux: float
    c_aux: float
    R_t: float
    action: str

    def to_dict(self) -> dict[str, Any]:
        """Return the required prompt fusion debug JSON schema."""

        return {
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "target_type": self.target_type,
            "B_sam": self.B_sam,
            "B_aux": self.B_aux,
            "IoU_aux": self.IoU_aux,
            "c_aux": self.c_aux,
            "R_t": self.R_t,
            "action": self.action,
        }


@dataclass(slots=True)
class FusedPrompt:
    """Corrective prompt returned to a SAM2/SAM3 propagation loop."""

    action: str
    box_prompt: list[float] | None
    positive_point: list[float] | None
    negative_points: list[list[float]]
    reason: str
    target_type: str
    debug_record: PromptFusionDebugRecord
    selected_candidate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable fused prompt metadata."""

        payload = asdict(self)
        payload["debug_record"] = self.debug_record.to_dict()
        return payload


def _as_config(config: PromptFusionConfig | dict[str, Any] | None) -> PromptFusionConfig:
    """Normalize prompt fusion config input."""

    if config is None:
        return PromptFusionConfig()
    if isinstance(config, PromptFusionConfig):
        return config
    allowed = PromptFusionConfig.__dataclass_fields__.keys()
    return PromptFusionConfig(**{key: value for key, value in config.items() if key in allowed})


def _read_field(item: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a dictionary or object."""

    if item is None:
        return default
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _normalize_box(box: BBox) -> list[float] | None:
    """Normalize a box-like object to ``[xmin, ymin, xmax, ymax]`` floats."""

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


def _box_area(box: list[float] | None) -> float:
    """Return inclusive bbox area."""

    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0] + 1.0) * max(0.0, box[3] - box[1] + 1.0)


def _box_center(box: list[float] | None) -> list[float] | None:
    """Return bbox center as an ``[x, y]`` point."""

    if box is None:
        return None
    return [float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0)]


def _candidate_iter(items: Iterable[Any] | None) -> list[Any]:
    """Return a list from an optional candidate iterable."""

    if items is None:
        return []
    return list(items)


def _candidate_box(candidate: Any) -> list[float] | None:
    """Read a candidate bbox from common anchor/detector field names."""

    return _normalize_box(_read_field(candidate, "bbox", "B_aux", "box", "target_bbox", default=None))


def _candidate_similarity(candidate: Any) -> float:
    """Read a candidate appearance similarity from common field names."""

    value = _read_field(candidate, "S_app", "top1_sim", "similarity", "score", "q", default=0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_confidence(candidate: Any) -> float:
    """Read candidate confidence from common detector/anchor fields."""

    value = _read_field(candidate, "confidence", "q", "score", "S_app", default=None)
    if value is None:
        return _candidate_similarity(candidate)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_to_dict(candidate: Any) -> dict[str, Any]:
    """Convert a candidate-like object to a compact JSON-safe dictionary."""

    if isinstance(candidate, dict):
        payload = dict(candidate)
    elif hasattr(candidate, "to_dict"):
        payload = dict(candidate.to_dict())
    elif hasattr(candidate, "__dict__"):
        payload = dict(candidate.__dict__)
    else:
        payload = {"repr": repr(candidate)}
    box = _candidate_box(candidate)
    if box is not None:
        payload["bbox"] = box
    if "S_app" not in payload:
        payload["S_app"] = _candidate_similarity(candidate)
    return payload


def _best_semantic_candidate(candidates: Iterable[Any] | None) -> tuple[Any | None, list[float] | None, float]:
    """Return the highest-appearance candidate and its bbox/similarity."""

    best_candidate: Any | None = None
    best_box: list[float] | None = None
    best_similarity = float("-inf")
    for candidate in _candidate_iter(candidates):
        box = _candidate_box(candidate)
        if box is None:
            continue
        similarity = _candidate_similarity(candidate)
        if similarity > best_similarity:
            best_candidate = candidate
            best_box = box
            best_similarity = similarity
    if best_candidate is None:
        return None, None, 0.0
    return best_candidate, best_box, float(best_similarity)


def _negative_points_from_distractors(
    candidates: Iterable[Any] | None,
    selected_box: list[float] | None,
    config: PromptFusionConfig,
) -> list[list[float]]:
    """Create negative point prompts from high-similarity distractor candidates."""

    points: list[list[float]] = []
    seen_centers: set[tuple[int, int]] = set()
    ranked: list[tuple[float, list[float]]] = []
    for candidate in _candidate_iter(candidates):
        box = _candidate_box(candidate)
        if box is None:
            continue
        similarity = _candidate_similarity(candidate)
        if similarity < config.distractor_similarity_threshold:
            continue
        if selected_box is not None and bbox_iou(box, selected_box) > config.negative_exclude_iou:
            continue
        center = _box_center(box)
        if center is not None:
            center_key = (int(round(center[0] * 10)), int(round(center[1] * 10)))
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)
            ranked.append((similarity, center))
    for _, center in sorted(ranked, key=lambda item: item[0], reverse=True)[: config.max_negative_points]:
        points.append(center)
    return points


def _target_type(target_profile: Any) -> str:
    """Read target type from a TargetProfile or JSON-like profile."""

    return str(_read_field(target_profile, "target_type", default="regular"))


def _target_bool(target_profile: Any, name: str) -> bool:
    """Read a boolean target profile flag."""

    return bool(_read_field(target_profile, name, default=False))


def _target_area(target_profile: Any) -> float | None:
    """Read expected target area from a TargetProfile or dictionary."""

    value = _read_field(target_profile, "area", default=None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_prompt(
    inputs: FramePromptInputs,
    target_type: str,
    box_sam: list[float] | None,
    selected_box: list[float] | None,
    selected_confidence: float,
    iou_aux: float,
    action: str,
    reason: str,
    selected_candidate: Any | None,
    config: PromptFusionConfig,
) -> FusedPrompt:
    """Build a fused prompt and its debug record."""

    previous_good_box = _normalize_box(inputs.previous_good_bbox)
    positive_source = previous_good_box if previous_good_box is not None and selected_box is not None else selected_box
    debug = PromptFusionDebugRecord(
        frame_id=inputs.frame_id,
        object_id=inputs.object_id,
        target_type=target_type,
        B_sam=box_sam,
        B_aux=selected_box,
        IoU_aux=float(iou_aux),
        c_aux=float(selected_confidence),
        R_t=float(inputs.R_t),
        action=action,
    )
    distractor_candidates = _candidate_iter(inputs.distractor_candidates) + _candidate_iter(inputs.candidates)
    return FusedPrompt(
        action=action,
        box_prompt=selected_box,
        positive_point=_box_center(positive_source),
        negative_points=_negative_points_from_distractors(distractor_candidates, selected_box, config),
        reason=reason,
        target_type=target_type,
        debug_record=debug,
        selected_candidate=_candidate_to_dict(selected_candidate) if selected_candidate is not None else None,
    )


def fuse_prompt_for_frame(
    inputs: FramePromptInputs,
    target_profile: Any,
    config: PromptFusionConfig | dict[str, Any] | None = None,
) -> FusedPrompt:
    """Fuse SAM and auxiliary tracker/detector signals into a corrective prompt."""

    cfg = _as_config(config)
    box_sam = _normalize_box(inputs.B_sam)
    box_aux = _normalize_box(inputs.B_aux)
    target_type = _target_type(target_profile)
    is_tiny = _target_bool(target_profile, "tiny") or target_type == "tiny"
    is_semantic = _target_bool(target_profile, "semantic_dominated") or target_type == "semantic_dominated"
    aux_confidence = float(inputs.c_aux or 0.0)
    iou_aux = bbox_iou(box_sam, box_aux) if box_sam is not None and box_aux is not None else 0.0
    aux_valid = box_aux is not None and aux_confidence > cfg.tau_conf
    generic_correction = bool(aux_valid and iou_aux < cfg.tau_iou and float(inputs.R_t) < cfg.tau_R)

    if is_semantic:
        candidate, candidate_box, candidate_similarity = _best_semantic_candidate(inputs.candidates)
        if candidate_box is not None:
            return _build_prompt(
                inputs,
                "semantic_dominated",
                box_sam,
                candidate_box,
                candidate_similarity,
                bbox_iou(box_sam, candidate_box) if box_sam is not None else 0.0,
                "use_semantic_candidate_box",
                "semantic_dominated_best_similarity",
                candidate,
                cfg,
            )
        if generic_correction:
            return _build_prompt(
                inputs,
                "semantic_dominated",
                box_sam,
                box_aux,
                aux_confidence,
                iou_aux,
                "use_aux_box",
                "generic_low_iou_low_reliability",
                None,
                cfg,
            )

    expected_area = inputs.expected_area if inputs.expected_area is not None else _target_area(target_profile)
    sam_area = float(inputs.sam_area) if inputs.sam_area is not None else _box_area(box_sam)
    tiny_area_drop = bool(expected_area is not None and sam_area < cfg.tiny_area_drop_ratio * max(1.0, float(expected_area)))
    if is_tiny:
        if aux_valid and tiny_area_drop:
            return _build_prompt(
                inputs,
                "tiny",
                box_sam,
                box_aux,
                aux_confidence,
                iou_aux,
                "use_tracker_box_tiny_area_drop",
                "tiny_area_collapse",
                None,
                cfg,
            )
        if aux_valid and (generic_correction or float(inputs.R_t) < cfg.tau_R):
            return _build_prompt(
                inputs,
                "tiny",
                box_sam,
                box_aux,
                aux_confidence,
                iou_aux,
                "use_tracker_box_tiny",
                "tiny_prefers_tracker_aux",
                None,
                cfg,
            )

    regular_trigger = bool(float(inputs.R_t) < cfg.regular_reliability_threshold or float(inputs.area_jump) > cfg.regular_area_jump_threshold)
    if not is_tiny and not is_semantic and aux_valid and regular_trigger:
        if iou_aux < cfg.tau_iou or float(inputs.area_jump) > cfg.regular_area_jump_threshold:
            return _build_prompt(
                inputs,
                "regular",
                box_sam,
                box_aux,
                aux_confidence,
                iou_aux,
                "use_aux_box_regular",
                "regular_low_reliability_or_area_jump",
                None,
                cfg,
            )

    if not is_tiny and not is_semantic and generic_correction and regular_trigger:
        return _build_prompt(
            inputs,
            "regular",
            box_sam,
            box_aux,
            aux_confidence,
            iou_aux,
            "use_aux_box",
            "generic_low_iou_low_reliability",
            None,
            cfg,
        )

    return _build_prompt(
        inputs,
        target_type,
        box_sam,
        None,
        aux_confidence,
        iou_aux,
        "keep_sam",
        "no_correction_triggered",
        None,
        cfg,
    )


def save_prompt_fusion_debug(records: Iterable[PromptFusionDebugRecord | FusedPrompt | dict[str, Any]], output_path: str | Path) -> None:
    """Write prompt fusion debug records to ``prompt_fusion_debug.json``."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, FusedPrompt):
            payload.append(record.debug_record.to_dict())
        elif isinstance(record, PromptFusionDebugRecord):
            payload.append(record.to_dict())
        elif isinstance(record, dict):
            payload.append({key: record.get(key) for key in PromptFusionDebugRecord.__dataclass_fields__.keys()})
        else:
            raise TypeError(f"Unsupported prompt fusion debug record type: {type(record).__name__}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

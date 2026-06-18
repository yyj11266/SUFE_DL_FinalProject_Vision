"""Reliability-aware tracking state machine and memory governance."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from src.vos.reliability import bbox_iou, mask_area, mask_to_bbox


BBox = list[float] | tuple[float, float, float, float] | np.ndarray | None
MaskRef = np.ndarray | Image.Image | str | Path | None
FeatureRef = np.ndarray | Sequence[float] | str | Path | None


@dataclass(slots=True)
class MemoryGovernanceConfig:
    """Thresholds and limits for reliability-aware memory governance."""

    stable_to_ambiguous_reliability: float = 0.65
    stable_to_ambiguous_area_jump: float = 1.0
    stable_to_ambiguous_motion_iou: float = 0.25
    ambiguous_to_recovery_frames: int = 3
    ambiguous_to_recovery_reliability: float = 0.40
    ambiguous_to_stable_frames: int = 2
    ambiguous_to_stable_reliability: float = 0.70
    recovery_to_stable_reliability: float = 0.72
    recovery_to_stable_appearance: float = 0.70
    max_recovery_frames: int = 5
    anchor_reliability_threshold: float = 0.78
    anchor_appearance_threshold: float = 0.72
    max_anchors: int = 8
    temporal_nms: int = 8
    delayed_promotion_reliability: float = 0.75
    delayed_promotion_horizon: int = 2
    recent_stable_anchors: int = 3
    selected_anchor_count: int = 4


@dataclass(slots=True)
class MemoryAnchor:
    """A first-frame or high-confidence stable memory anchor."""

    frame_id: str | int
    frame_index: int
    object_id: str | int
    mask: MaskRef = None
    bbox: list[float] | None = None
    feature: FeatureRef = None
    reliability: float = 1.0
    appearance_similarity: float = 1.0
    source: str = "stable_anchor"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe anchor summary."""

        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "mask": _mask_ref(self.mask),
            "bbox": _json_box(self.bbox),
            "feature": _feature_ref(self.feature),
            "reliability": float(self.reliability),
            "appearance_similarity": float(self.appearance_similarity),
            "source": self.source,
            "metadata": _json_safe_value(self.metadata),
        }


@dataclass(slots=True)
class ObjectTrackState:
    """Mutable object-level state used by memory governance."""

    object_id: str | int
    state: str = "stable"
    consecutive_ambiguous: int = 0
    consecutive_recovery: int = 0
    consecutive_stable: int = 0
    ambiguous_stable_streak: int = 0
    last_reliable_frame: int | None = None
    first_frame_anchor: MemoryAnchor | None = None
    anchor_bank: list[MemoryAnchor] = field(default_factory=list)
    pending_promotions: list[MemoryAnchor] = field(default_factory=list)
    strong_memory_frames: list[str | int] = field(default_factory=list)
    weak_memory_frames: list[str | int] = field(default_factory=list)
    recent_stable_memory: list[MemoryAnchor] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe track state for debug output."""

        return {
            "object_id": self.object_id,
            "state": self.state,
            "consecutive_ambiguous": self.consecutive_ambiguous,
            "consecutive_recovery": self.consecutive_recovery,
            "consecutive_stable": self.consecutive_stable,
            "ambiguous_stable_streak": self.ambiguous_stable_streak,
            "last_reliable_frame": self.last_reliable_frame,
            "first_frame_anchor": self.first_frame_anchor.to_dict() if self.first_frame_anchor else None,
            "anchor_bank": [anchor.to_dict() for anchor in self.anchor_bank],
            "pending_promotions": [anchor.to_dict() for anchor in self.pending_promotions],
            "strong_memory_frames": self.strong_memory_frames,
            "weak_memory_frames": self.weak_memory_frames,
            "recent_stable_memory": [anchor.to_dict() for anchor in self.recent_stable_memory],
            "history": self.history,
        }


@dataclass(slots=True)
class MemoryDecision:
    """Governance decision emitted for one object on one frame."""

    frame_id: str | int
    frame_index: int
    object_id: str | int
    previous_state: str
    new_state: str
    transition_reason: str
    memory_write_policy: str
    allow_branching: bool
    allow_empty_mask: bool
    selected_memory_anchors: list[MemoryAnchor]
    promoted_frames: list[str | int]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision summary."""

        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "object_id": self.object_id,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "transition_reason": self.transition_reason,
            "memory_write_policy": self.memory_write_policy,
            "allow_branching": self.allow_branching,
            "allow_empty_mask": self.allow_empty_mask,
            "selected_memory_anchors": [anchor.to_dict() for anchor in self.selected_memory_anchors],
            "promoted_frames": self.promoted_frames,
            "debug": _json_safe_value(self.debug),
        }


def initialize_object_track_state(
    object_id: str | int,
    first_frame_id: str | int,
    first_frame_index: int = 0,
    mask: MaskRef = None,
    bbox: BBox = None,
    feature: FeatureRef = None,
    metadata: dict[str, Any] | None = None,
) -> ObjectTrackState:
    """Create an object track state with a permanent first-frame anchor."""

    anchor = MemoryAnchor(
        frame_id=first_frame_id,
        frame_index=int(first_frame_index),
        object_id=object_id,
        mask=mask,
        bbox=_json_box(bbox if bbox is not None else _bbox_from_mask(mask)),
        feature=feature,
        reliability=1.0,
        appearance_similarity=1.0,
        source="first_frame",
        metadata=dict(metadata or {}),
    )
    return ObjectTrackState(
        object_id=object_id,
        state="stable",
        consecutive_stable=1,
        last_reliable_frame=int(first_frame_index),
        first_frame_anchor=anchor,
        anchor_bank=[anchor],
        recent_stable_memory=[anchor],
    )


def update_tracking_state(
    track_state: ObjectTrackState,
    reliability_result: Any,
    bbox_t: BBox = None,
    motion_bbox: BBox = None,
    candidate_branches: Iterable[Any] | None = None,
    config: MemoryGovernanceConfig | dict[str, Any] | None = None,
) -> MemoryDecision:
    """Update one object's state machine and emit memory governance policy."""

    cfg = _as_config(config)
    event = _event_from_reliability(reliability_result, bbox_t, motion_bbox)
    _ensure_first_anchor(track_state, event)
    previous_state = track_state.state
    branch = _best_recovery_branch(candidate_branches, cfg)
    transition = _transition_state(track_state, event, branch, cfg)
    _apply_state_counters(track_state, transition["new_state"], event, cfg)
    current_anchor = _anchor_from_event(event, source="current_frame")
    promoted_frames, promotion_debug = _update_pending_promotions(track_state, current_anchor, event, cfg)
    anchor_debug = _maybe_add_anchor(track_state, current_anchor, event, cfg)
    selected_anchors = _select_memory_anchors(track_state, event, cfg)
    memory_policy = _memory_write_policy(track_state.state, event, cfg)
    allow_branching = track_state.state in {"ambiguous", "recovery"}
    allow_empty = track_state.state == "lost"
    if track_state.state == "stable" and current_anchor is not None:
        _append_recent_stable(track_state, current_anchor, cfg)
    if event["reliability"] >= cfg.stable_to_ambiguous_reliability and track_state.state == "stable":
        track_state.last_reliable_frame = event["frame_index"]
    decision = MemoryDecision(
        frame_id=event["frame_id"],
        frame_index=event["frame_index"],
        object_id=track_state.object_id,
        previous_state=previous_state,
        new_state=track_state.state,
        transition_reason=transition["reason"],
        memory_write_policy=memory_policy,
        allow_branching=allow_branching,
        allow_empty_mask=allow_empty,
        selected_memory_anchors=selected_anchors,
        promoted_frames=promoted_frames,
        debug={
            "signals": transition["signals"],
            "candidate_branch": branch,
            "counters": {
                "consecutive_ambiguous": track_state.consecutive_ambiguous,
                "consecutive_recovery": track_state.consecutive_recovery,
                "consecutive_stable": track_state.consecutive_stable,
                "ambiguous_stable_streak": track_state.ambiguous_stable_streak,
            },
            "anchor_bank_size": len(track_state.anchor_bank),
            "pending_promotions": [anchor.frame_id for anchor in track_state.pending_promotions],
            "promotion": promotion_debug,
            "anchor_update": anchor_debug,
        },
    )
    track_state.history.append(decision.to_dict())
    return decision


def save_memory_governance_debug(
    track_state: ObjectTrackState,
    decisions: Iterable[MemoryDecision | dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write memory governance state and decisions to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "track_state": track_state.to_dict(),
        "decisions": [decision.to_dict() if isinstance(decision, MemoryDecision) else decision for decision in decisions],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _as_config(config: MemoryGovernanceConfig | dict[str, Any] | None) -> MemoryGovernanceConfig:
    """Normalize config input."""

    if config is None:
        return MemoryGovernanceConfig()
    if isinstance(config, MemoryGovernanceConfig):
        return config
    allowed = MemoryGovernanceConfig.__dataclass_fields__.keys()
    return MemoryGovernanceConfig(**{key: value for key, value in config.items() if key in allowed})


def _read_field(item: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a dictionary or object."""

    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _event_from_reliability(reliability_result: Any, bbox_t: BBox, motion_bbox: BBox) -> dict[str, Any]:
    """Extract the governance event fields from a ReliabilityResult-like object."""

    frame_id = _read_field(reliability_result, "frame_id", default="")
    frame_index = _read_field(reliability_result, "frame_index", default=None)
    if frame_index is None:
        frame_index = _frame_index_from_id(frame_id)
    bbox = _json_box(bbox_t if bbox_t is not None else _read_field(reliability_result, "current_bbox", "bbox", default=None))
    motion = _json_box(motion_bbox if motion_bbox is not None else _read_field(reliability_result, "motion_pred_bbox", "motion_bbox", default=None))
    motion_iou = bbox_iou(bbox, motion) if bbox is not None and motion is not None else _optional_float(_read_field(reliability_result, "motion_consistency", default=1.0), 1.0)
    return {
        "frame_id": frame_id,
        "frame_index": int(frame_index),
        "object_id": _read_field(reliability_result, "object_id", default=1),
        "reliability": _optional_float(_read_field(reliability_result, "reliability", "R_t", default=0.0), 0.0),
        "area_jump": _optional_float(_read_field(reliability_result, "area_jump", default=0.0), 0.0),
        "appearance_similarity": _optional_float(_read_field(reliability_result, "appearance_similarity", default=0.0), 0.0),
        "motion_iou": float(motion_iou),
        "bbox": bbox,
        "motion_bbox": motion,
        "mask": _read_field(reliability_result, "mask", "current_mask", "mask_t", default=None),
        "feature": _read_field(reliability_result, "feature", "feature_t", default=None),
        "metadata": dict(_read_field(reliability_result, "metadata", default={}) or {}),
    }


def _frame_index_from_id(frame_id: str | int) -> int:
    """Infer a numeric frame index from a frame id."""

    if isinstance(frame_id, int):
        return int(frame_id)
    digits = "".join(char for char in str(frame_id) if char.isdigit())
    return int(digits) if digits else 0


def _optional_float(value: Any, default: float | None = None) -> float | None:
    """Convert a value to float, returning default on failure."""

    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _transition_state(
    track_state: ObjectTrackState,
    event: dict[str, Any],
    best_branch: dict[str, Any] | None,
    config: MemoryGovernanceConfig,
) -> dict[str, Any]:
    """Return the next state and transition diagnostics."""

    r = float(event["reliability"])
    area_jump = float(event["area_jump"])
    motion_iou = float(event["motion_iou"])
    signals = {
        "reliability": r,
        "area_jump": area_jump,
        "motion_iou": motion_iou,
        "appearance_similarity": float(event["appearance_similarity"]),
    }
    if track_state.state == "stable":
        reasons: list[str] = []
        if r < config.stable_to_ambiguous_reliability:
            reasons.append("low_reliability")
        if area_jump > config.stable_to_ambiguous_area_jump:
            reasons.append("area_jump")
        if motion_iou < config.stable_to_ambiguous_motion_iou:
            reasons.append("motion_inconsistent")
        if reasons:
            return {"new_state": "ambiguous", "reason": "stable_to_ambiguous:" + ",".join(reasons), "signals": signals}
        return {"new_state": "stable", "reason": "stable_remain", "signals": signals}
    if track_state.state == "ambiguous":
        if r < config.ambiguous_to_recovery_reliability:
            return {"new_state": "recovery", "reason": "ambiguous_to_recovery:very_low_reliability", "signals": signals}
        if r >= config.ambiguous_to_stable_reliability and track_state.ambiguous_stable_streak + 1 >= config.ambiguous_to_stable_frames:
            return {"new_state": "stable", "reason": "ambiguous_to_stable:stable_streak", "signals": signals}
        if r < config.ambiguous_to_stable_reliability and track_state.consecutive_ambiguous + 1 >= config.ambiguous_to_recovery_frames:
            return {"new_state": "recovery", "reason": "ambiguous_to_recovery:ambiguous_streak", "signals": signals}
        return {"new_state": "ambiguous", "reason": "ambiguous_remain", "signals": signals}
    if track_state.state == "recovery":
        if best_branch is not None:
            return {"new_state": "stable", "reason": "recovery_to_stable:reliable_candidate", "signals": signals}
        if track_state.consecutive_recovery + 1 >= config.max_recovery_frames:
            return {"new_state": "lost", "reason": "recovery_to_lost:max_recovery_frames", "signals": signals}
        return {"new_state": "recovery", "reason": "recovery_remain", "signals": signals}
    if track_state.state == "lost":
        return {"new_state": "lost", "reason": "lost_remain", "signals": signals}
    return {"new_state": "stable", "reason": f"unknown_previous_state:{track_state.state}", "signals": signals}


def _apply_state_counters(
    track_state: ObjectTrackState,
    new_state: str,
    event: dict[str, Any],
    config: MemoryGovernanceConfig,
) -> None:
    """Apply state transition and update consecutive-frame counters."""

    previous = track_state.state
    r = float(event["reliability"])
    track_state.state = new_state
    if new_state == "stable":
        track_state.consecutive_stable = track_state.consecutive_stable + 1 if previous == "stable" else 1
        track_state.consecutive_ambiguous = 0
        track_state.consecutive_recovery = 0
        track_state.ambiguous_stable_streak = 0
    elif new_state == "ambiguous":
        if previous == "ambiguous" and r >= config.ambiguous_to_stable_reliability:
            track_state.ambiguous_stable_streak += 1
        else:
            track_state.ambiguous_stable_streak = 0
        if previous == "ambiguous" and r < config.ambiguous_to_stable_reliability:
            track_state.consecutive_ambiguous += 1
        elif previous == "ambiguous":
            track_state.consecutive_ambiguous = max(0, track_state.consecutive_ambiguous)
        else:
            track_state.consecutive_ambiguous = 1
        track_state.consecutive_stable = 0
        track_state.consecutive_recovery = 0
    elif new_state == "recovery":
        track_state.consecutive_recovery = track_state.consecutive_recovery + 1 if previous == "recovery" else 1
        track_state.consecutive_ambiguous = 0
        track_state.consecutive_stable = 0
        track_state.ambiguous_stable_streak = 0
    else:
        track_state.consecutive_ambiguous = 0
        track_state.consecutive_stable = 0
        track_state.ambiguous_stable_streak = 0


def _best_recovery_branch(candidate_branches: Iterable[Any] | None, config: MemoryGovernanceConfig) -> dict[str, Any] | None:
    """Return the best reliable recovery branch if one exists."""

    best: dict[str, Any] | None = None
    for branch in candidate_branches or []:
        reliability = _optional_float(_read_field(branch, "reliability", "R_t", default=0.0), 0.0)
        appearance = _optional_float(_read_field(branch, "appearance_similarity", "S_app", default=0.0), 0.0)
        if reliability >= config.recovery_to_stable_reliability and appearance >= config.recovery_to_stable_appearance:
            payload = _branch_to_debug(branch)
            payload["reliability"] = reliability
            payload["appearance_similarity"] = appearance
            if best is None or reliability + appearance > best["reliability"] + best["appearance_similarity"]:
                best = payload
    return best


def _branch_to_debug(branch: Any) -> dict[str, Any]:
    """Return compact JSON-safe branch debug metadata."""

    if isinstance(branch, dict):
        payload = dict(branch)
    elif hasattr(branch, "to_dict"):
        payload = branch.to_dict()
    else:
        payload = {key: _read_field(branch, key, default=None) for key in ("frame_id", "source", "bbox", "reliability", "appearance_similarity")}
    return _json_safe_value(payload)


def _ensure_first_anchor(track_state: ObjectTrackState, event: dict[str, Any]) -> None:
    """Create a permanent first-frame anchor if state was constructed empty."""

    if track_state.first_frame_anchor is not None:
        if not any(anchor.source == "first_frame" for anchor in track_state.anchor_bank):
            track_state.anchor_bank.insert(0, track_state.first_frame_anchor)
        return
    anchor = MemoryAnchor(
        frame_id=event["frame_id"],
        frame_index=event["frame_index"],
        object_id=track_state.object_id,
        mask=event.get("mask"),
        bbox=_json_box(event.get("bbox") or _bbox_from_mask(event.get("mask"))),
        feature=event.get("feature"),
        reliability=1.0,
        appearance_similarity=1.0,
        source="first_frame",
        metadata={"auto_initialized": True, **dict(event.get("metadata", {}))},
    )
    track_state.first_frame_anchor = anchor
    track_state.anchor_bank.insert(0, anchor)


def _anchor_from_event(event: dict[str, Any], source: str) -> MemoryAnchor:
    """Create an anchor-like object from the current event."""

    return MemoryAnchor(
        frame_id=event["frame_id"],
        frame_index=event["frame_index"],
        object_id=event["object_id"],
        mask=event.get("mask"),
        bbox=_json_box(event.get("bbox") or _bbox_from_mask(event.get("mask"))),
        feature=event.get("feature"),
        reliability=float(event["reliability"]),
        appearance_similarity=float(event["appearance_similarity"]),
        source=source,
        metadata=dict(event.get("metadata", {})),
    )


def _update_pending_promotions(
    track_state: ObjectTrackState,
    current_anchor: MemoryAnchor,
    event: dict[str, Any],
    config: MemoryGovernanceConfig,
) -> tuple[list[str | int], dict[str, Any]]:
    """Advance delayed-promotion bookkeeping."""

    promoted: list[str | int] = []
    discarded: list[str | int] = []
    if track_state.state != "stable":
        discarded = [anchor.frame_id for anchor in track_state.pending_promotions]
        track_state.pending_promotions = []
        return promoted, {"discarded": discarded, "added_pending": None}
    kept: list[MemoryAnchor] = []
    for anchor in track_state.pending_promotions:
        if anchor.frame_index < event["frame_index"]:
            anchor.metadata["stable_followup_count"] = int(anchor.metadata.get("stable_followup_count", 0)) + 1
        if int(anchor.metadata.get("stable_followup_count", 0)) >= config.delayed_promotion_horizon:
            if anchor.frame_id not in track_state.strong_memory_frames:
                track_state.strong_memory_frames.append(anchor.frame_id)
            promoted.append(anchor.frame_id)
        else:
            kept.append(anchor)
    track_state.pending_promotions = kept
    added_pending: str | int | None = None
    is_first_anchor = track_state.first_frame_anchor is not None and str(current_anchor.frame_id) == str(track_state.first_frame_anchor.frame_id)
    if current_anchor.reliability >= config.delayed_promotion_reliability and current_anchor.source != "first_frame" and not is_first_anchor:
        if current_anchor.frame_id not in {anchor.frame_id for anchor in track_state.pending_promotions}:
            pending_anchor = _copy_anchor(current_anchor, source=current_anchor.source)
            pending_anchor.metadata["stable_followup_count"] = 0
            track_state.pending_promotions.append(pending_anchor)
            added_pending = current_anchor.frame_id
    if current_anchor.frame_id not in track_state.weak_memory_frames:
        track_state.weak_memory_frames.append(current_anchor.frame_id)
    return promoted, {"discarded": discarded, "added_pending": added_pending}


def _maybe_add_anchor(
    track_state: ObjectTrackState,
    current_anchor: MemoryAnchor,
    event: dict[str, Any],
    config: MemoryGovernanceConfig,
) -> dict[str, Any]:
    """Add high-confidence stable anchors to the anchor bank when eligible."""

    if track_state.state != "stable":
        return {"added": False, "reason": "state_not_stable"}
    if current_anchor.reliability < config.anchor_reliability_threshold:
        return {"added": False, "reason": "low_reliability"}
    if current_anchor.appearance_similarity < config.anchor_appearance_threshold:
        return {"added": False, "reason": "low_appearance"}
    if any(abs(current_anchor.frame_index - anchor.frame_index) < config.temporal_nms for anchor in track_state.anchor_bank):
        return {"added": False, "reason": "temporal_nms"}
    bank_anchor = _copy_anchor(current_anchor, source="stable_anchor")
    bank_anchor.metadata.pop("stable_followup_count", None)
    track_state.anchor_bank.append(bank_anchor)
    _prune_anchor_bank(track_state, config)
    return {"added": True, "frame_id": current_anchor.frame_id}


def _prune_anchor_bank(track_state: ObjectTrackState, config: MemoryGovernanceConfig) -> None:
    """Prune anchor bank while preserving the first-frame anchor."""

    first = track_state.first_frame_anchor
    anchors = [anchor for anchor in track_state.anchor_bank if first is None or anchor.frame_id != first.frame_id]
    anchors = sorted(anchors, key=lambda anchor: (anchor.reliability + anchor.appearance_similarity, anchor.frame_index), reverse=True)
    kept: list[MemoryAnchor] = []
    for anchor in anchors:
        if len(kept) >= max(0, config.max_anchors - (1 if first is not None else 0)):
            break
        if all(abs(anchor.frame_index - other.frame_index) >= config.temporal_nms for other in kept):
            kept.append(anchor)
    track_state.anchor_bank = ([first] if first is not None else []) + kept


def _append_recent_stable(track_state: ObjectTrackState, anchor: MemoryAnchor, config: MemoryGovernanceConfig) -> None:
    """Keep a short queue of recent stable anchors."""

    track_state.recent_stable_memory = [item for item in track_state.recent_stable_memory if item.frame_id != anchor.frame_id]
    track_state.recent_stable_memory.append(anchor)
    track_state.recent_stable_memory = track_state.recent_stable_memory[-config.recent_stable_anchors :]


def _select_memory_anchors(
    track_state: ObjectTrackState,
    event: dict[str, Any],
    config: MemoryGovernanceConfig,
) -> list[MemoryAnchor]:
    """Select anchors according to current memory policy."""

    first = [track_state.first_frame_anchor] if track_state.first_frame_anchor is not None else []
    if track_state.state == "stable":
        candidates = first + track_state.recent_stable_memory + _rank_anchor_bank(track_state.anchor_bank, event, config)
    elif track_state.state == "ambiguous":
        candidates = first + _rank_anchor_bank(track_state.anchor_bank, event, config)
    elif track_state.state == "recovery":
        candidates = first + _rank_anchor_bank(track_state.anchor_bank, event, config)
    elif track_state.state == "lost":
        candidates = first
    else:
        candidates = first
    return _dedupe_anchors(candidates)[: config.selected_anchor_count]


def _rank_anchor_bank(
    anchors: list[MemoryAnchor],
    event: dict[str, Any],
    config: MemoryGovernanceConfig,
) -> list[MemoryAnchor]:
    """Rank anchor bank by temporal proximity and optional feature similarity."""

    current_feature = event.get("feature")
    current_index = int(event["frame_index"])

    def score(anchor: MemoryAnchor) -> float:
        temporal = 1.0 / (1.0 + abs(current_index - anchor.frame_index))
        feature_sim = _feature_similarity(current_feature, anchor.feature)
        feature_score = feature_sim if feature_sim is not None else 0.5 * (anchor.reliability + anchor.appearance_similarity)
        return 0.55 * feature_score + 0.45 * temporal

    return sorted(anchors, key=score, reverse=True)


def _dedupe_anchors(anchors: Iterable[MemoryAnchor | None]) -> list[MemoryAnchor]:
    """Dedupe anchors by frame id while preserving order."""

    seen: set[str] = set()
    output: list[MemoryAnchor] = []
    for anchor in anchors:
        if anchor is None:
            continue
        key = str(anchor.frame_id)
        if key in seen:
            continue
        seen.add(key)
        output.append(anchor)
    return output


def _copy_anchor(anchor: MemoryAnchor, source: str | None = None) -> MemoryAnchor:
    """Return a shallow copy of an anchor with isolated metadata."""

    return MemoryAnchor(
        frame_id=anchor.frame_id,
        frame_index=anchor.frame_index,
        object_id=anchor.object_id,
        mask=anchor.mask,
        bbox=list(anchor.bbox) if anchor.bbox is not None else None,
        feature=anchor.feature,
        reliability=anchor.reliability,
        appearance_similarity=anchor.appearance_similarity,
        source=source or anchor.source,
        metadata=dict(anchor.metadata),
    )


def _memory_write_policy(state: str, event: dict[str, Any], config: MemoryGovernanceConfig) -> str:
    """Return outer memory write policy for the current state."""

    if state == "stable":
        return "weak_pending_promotion" if event["reliability"] >= config.delayed_promotion_reliability else "weak_non_conditioning"
    if state == "ambiguous":
        return "no_update_branch_candidates"
    if state == "recovery":
        return "recovery_reprompt_no_update"
    if state == "lost":
        return "no_update_allow_empty"
    return "no_update"


def _bbox_from_mask(mask: MaskRef) -> list[int] | None:
    """Infer a bbox from a mask reference when possible."""

    if mask is None:
        return None
    try:
        return mask_to_bbox(mask)
    except Exception:
        return None


def _json_box(box: BBox) -> list[float] | None:
    """Return a JSON-safe bbox."""

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


def _mask_ref(mask: MaskRef) -> str | dict[str, Any] | None:
    """Summarize a mask reference without serializing heavy arrays."""

    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return str(mask)
    if isinstance(mask, Image.Image):
        return {"type": "PIL.Image", "size": list(mask.size)}
    array = np.asarray(mask)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype), "area": int(mask_area(array))}


def _feature_ref(feature: FeatureRef) -> str | dict[str, Any] | None:
    """Summarize a feature reference without serializing heavy vectors."""

    if feature is None:
        return None
    if isinstance(feature, (str, Path)):
        return str(feature)
    array = np.asarray(feature)
    return {"type": "feature", "shape": list(array.shape), "dtype": str(array.dtype)}


def _feature_similarity(feature_a: FeatureRef, feature_b: FeatureRef) -> float | None:
    """Return cosine similarity for numeric feature vectors when available."""

    if feature_a is None or feature_b is None or isinstance(feature_a, (str, Path)) or isinstance(feature_b, (str, Path)):
        return None
    try:
        a = np.asarray(feature_a, dtype=np.float32).reshape(-1)
        b = np.asarray(feature_b, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return None
    return float(np.dot(a, b) / denom)


def _json_safe_value(value: Any) -> Any:
    """Convert common numpy/path/dataclass values to JSON-safe primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, MemoryAnchor):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe_value(asdict(value))
    return str(value)


def _load_events_json(path: str | Path) -> list[dict[str, Any]]:
    """Load CLI events JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    if isinstance(payload, dict):
        payload = [{"frame_id": frame_id, **event} for frame_id, event in payload.items()]
    if not isinstance(payload, list):
        raise ValueError("events JSON must be a list, mapping, or contain an 'events' list")
    return [dict(event) for event in payload]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for state-machine replay."""

    parser = argparse.ArgumentParser(description="Replay reliability-aware memory governance decisions.")
    parser.add_argument("--events-json", required=True, help="Reliability event JSON list or mapping.")
    parser.add_argument("--output-json", required=True, help="Output memory_governance_debug.json path.")
    parser.add_argument("--object-id", required=True, help="Object id to replay.")
    parser.add_argument("--max-recovery-frames", type=int, default=5)
    parser.add_argument("--temporal-nms", type=int, default=8)
    parser.add_argument("--max-anchors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run memory governance CLI replay."""

    args = _parse_args()
    config = MemoryGovernanceConfig(
        max_recovery_frames=args.max_recovery_frames,
        temporal_nms=args.temporal_nms,
        max_anchors=args.max_anchors,
    )
    events = [event for event in _load_events_json(args.events_json) if str(event.get("object_id", args.object_id)) == str(args.object_id)]
    if not events:
        raise ValueError(f"No events found for object_id={args.object_id}")
    first = events[0]
    state = initialize_object_track_state(
        object_id=args.object_id,
        first_frame_id=first.get("frame_id", 0),
        first_frame_index=int(first.get("frame_index", _frame_index_from_id(first.get("frame_id", 0)))),
        mask=first.get("mask"),
        bbox=first.get("bbox", first.get("current_bbox")),
        feature=first.get("feature"),
    )
    decisions: list[MemoryDecision] = []
    for event in events:
        branches = event.get("candidate_branches", event.get("branches", []))
        decisions.append(
            update_tracking_state(
                state,
                event,
                bbox_t=event.get("bbox", event.get("current_bbox")),
                motion_bbox=event.get("motion_bbox", event.get("motion_pred_bbox")),
                candidate_branches=branches,
                config=config,
            )
        )
    save_memory_governance_debug(state, decisions, args.output_json)
    print(json.dumps({"output_json": args.output_json, "final_state": state.state, "num_decisions": len(decisions)}, indent=2))


if __name__ == "__main__":
    main()

"""Multi-anchor bidirectional propagation and mask/logit fusion."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.vos.reliability import bbox_iou, mask_iou


MaskLike = np.ndarray | Image.Image | str | Path | None
LogitLike = np.ndarray | str | Path | None


@dataclass(slots=True)
class PropagationAnchor:
    """Object anchor used to launch one or more propagation passes."""

    anchor_id: str
    frame_id: str | int
    frame_index: int
    object_id: str | int
    mask: MaskLike
    logit: LogitLike = None
    reliability: float = 1.0
    source: str = "anchor"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe anchor summary."""

        return {
            "anchor_id": self.anchor_id,
            "frame_id": self.frame_id,
            "frame_index": int(self.frame_index),
            "object_id": self.object_id,
            "mask": _mask_ref(self.mask),
            "logit": _logit_ref(self.logit),
            "reliability": float(self.reliability),
            "source": self.source,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class AnchorPropagationResult:
    """Propagation output for one anchor and one temporal direction."""

    anchor: PropagationAnchor
    direction: str
    masks: dict[str | int, MaskLike]
    logits: dict[str | int, LogitLike] = field(default_factory=dict)
    frame_index_mapping: dict[str | int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe propagation result summary."""

        return {
            "anchor": self.anchor.to_dict(),
            "direction": self.direction,
            "masks": {str(key): _mask_ref(value) for key, value in self.masks.items()},
            "logits": {str(key): _logit_ref(value) for key, value in self.logits.items()},
            "frame_index_mapping": {str(key): int(value) for key, value in self.frame_index_mapping.items()},
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class MultiAnchorPropagationConfig:
    """Configuration for bidirectional propagation and fusion."""

    tau: float = 25.0
    eps: float = 1e-6
    theta: float = 0.0
    conflict_iou_threshold: float = 0.30
    low_reliability_threshold: float = 0.40
    debug_video_fps: float = 6.0
    write_outputs: bool = True
    copy_reversed_frames: bool = True


@dataclass(slots=True)
class MultiAnchorPropagationResult:
    """Result of multi-anchor propagation and fusion."""

    fused_masks: dict[str | int, np.ndarray]
    per_anchor_results: list[AnchorPropagationResult]
    fusion_debug: dict[str | int, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    output_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result summary."""

        return {
            "fused_masks": {str(key): _mask_ref(value) for key, value in self.fused_masks.items()},
            "per_anchor_results": [result.to_dict() for result in self.per_anchor_results],
            "fusion_debug": {str(key): _json_safe(value) for key, value in self.fusion_debug.items()},
            "warnings": self.warnings,
            "output_paths": self.output_paths,
        }


@dataclass(slots=True)
class _FrameItem:
    """Normalized video frame metadata."""

    frame_id: str
    frame_index: int
    original_index: int
    frame: Any
    path: str | None = None


def run_multi_anchor_bidirectional_propagation(
    first_frame_mask: MaskLike,
    anchor_masks: Iterable[Any],
    video_frames: Iterable[Any],
    tracker_backend: Any,
    reliability_scores: Mapping[Any, float] | None,
    output_dir: str | Path,
    object_id: str | int = 1,
    config: MultiAnchorPropagationConfig | dict[str, Any] | None = None,
) -> MultiAnchorPropagationResult:
    """Run multi-anchor forward/backward propagation and fuse masks."""

    cfg = _as_config(config)
    if tracker_backend is None:
        raise RuntimeError("tracker_backend is required; no pseudo propagation is generated.")
    output_path = Path(output_dir)
    frames = _normalize_frames(video_frames)
    if not frames:
        raise ValueError("video_frames is empty")
    anchors = _build_anchor_set(first_frame_mask, anchor_masks, frames, object_id, reliability_scores)
    warnings: list[str] = []
    per_anchor_results: list[AnchorPropagationResult] = []
    for anchor in anchors:
        forward = _run_direction(
            backend=tracker_backend,
            frames=frames,
            anchor=anchor,
            direction="forward",
            output_dir=output_path,
            config=cfg,
        )
        per_anchor_results.append(forward)
        warnings.extend(forward.warnings)
        if anchor.frame_index > 0:
            backward = _run_direction(
                backend=tracker_backend,
                frames=frames,
                anchor=anchor,
                direction="backward",
                output_dir=output_path,
                config=cfg,
            )
            per_anchor_results.append(backward)
            warnings.extend(backward.warnings)
    fused_masks, fusion_debug = _fuse_all_frames(frames, per_anchor_results, cfg)
    result = MultiAnchorPropagationResult(
        fused_masks=fused_masks,
        per_anchor_results=per_anchor_results,
        fusion_debug=fusion_debug,
        warnings=warnings,
    )
    if cfg.write_outputs:
        _write_outputs(result, frames, output_path, cfg)
    return result


def _as_config(config: MultiAnchorPropagationConfig | dict[str, Any] | None) -> MultiAnchorPropagationConfig:
    """Normalize config input."""

    if config is None:
        return MultiAnchorPropagationConfig()
    if isinstance(config, MultiAnchorPropagationConfig):
        return config
    allowed = MultiAnchorPropagationConfig.__dataclass_fields__.keys()
    return MultiAnchorPropagationConfig(**{key: value for key, value in config.items() if key in allowed})


def _natural_key(value: Any) -> list[int | str]:
    """Sort strings with embedded numbers in numeric order."""

    import re

    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _normalize_frames(video_frames: Iterable[Any]) -> list[_FrameItem]:
    """Normalize frame inputs to stable indexed frame metadata."""

    items = list(video_frames)
    if all(isinstance(item, (str, Path)) for item in items):
        items = sorted(items, key=_natural_key)
    frames: list[_FrameItem] = []
    for index, item in enumerate(items):
        frame_id = _read_field(item, "frame_id", "frame_stem", default=None)
        path = _read_field(item, "path", "image_path", "original_path", "inference_path", default=None)
        frame_payload = item
        if isinstance(item, (str, Path)):
            path = str(item)
            frame_id = Path(item).stem
        if frame_id is None:
            frame_id = f"{index:05d}"
        frames.append(
            _FrameItem(
                frame_id=str(frame_id),
                frame_index=index,
                original_index=index,
                frame=frame_payload,
                path=str(path) if path is not None else None,
            )
        )
    return frames


def _build_anchor_set(
    first_frame_mask: MaskLike,
    anchor_masks: Iterable[Any],
    frames: list[_FrameItem],
    object_id: str | int,
    reliability_scores: Mapping[Any, float] | None,
) -> list[PropagationAnchor]:
    """Build propagation anchors from first frame and anchor-mining outputs."""

    anchors = [
        PropagationAnchor(
            anchor_id=f"{object_id}_first",
            frame_id=frames[0].frame_id,
            frame_index=0,
            object_id=object_id,
            mask=first_frame_mask,
            logit=None,
            reliability=_score_lookup(reliability_scores, frames[0].frame_id, 0, 1.0),
            source="first_frame",
        )
    ]
    seen = {anchors[0].anchor_id}
    for offset, raw_anchor in enumerate(anchor_masks or []):
        anchor = _anchor_from_any(raw_anchor, frames, object_id, reliability_scores, offset)
        if anchor.anchor_id in seen:
            anchor.anchor_id = f"{anchor.anchor_id}_{offset}"
        seen.add(anchor.anchor_id)
        anchors.append(anchor)
    return anchors


def _anchor_from_any(
    raw_anchor: Any,
    frames: list[_FrameItem],
    object_id: str | int,
    reliability_scores: Mapping[Any, float] | None,
    offset: int,
) -> PropagationAnchor:
    """Normalize anchor-like objects and dictionaries."""

    if isinstance(raw_anchor, PropagationAnchor):
        return raw_anchor
    frame_id = _read_field(raw_anchor, "frame_id", default=None)
    frame_index = _read_field(raw_anchor, "frame_index", default=None)
    if frame_index is None:
        frame_index = _frame_index_from_id(frame_id, frames)
    frame_index = int(frame_index)
    if frame_id is None and 0 <= frame_index < len(frames):
        frame_id = frames[frame_index].frame_id
    mask = _read_field(raw_anchor, "mask", "mask_path", default=None)
    logit = _read_field(raw_anchor, "logit", "logit_path", default=None)
    reliability = _read_field(raw_anchor, "reliability", "R_i", "R", default=None)
    if reliability is None:
        reliability = _score_lookup(reliability_scores, frame_id, frame_index, None)
    if reliability is None:
        reliability = _read_field(raw_anchor, "S_anchor", "S_app", "q", default=0.75)
    anchor_id = str(_read_field(raw_anchor, "anchor_id", default=f"{object_id}_anchor_{frame_index:05d}_{offset}"))
    return PropagationAnchor(
        anchor_id=anchor_id,
        frame_id=str(frame_id),
        frame_index=frame_index,
        object_id=_read_field(raw_anchor, "object_id", default=object_id),
        mask=mask,
        logit=logit,
        reliability=float(reliability),
        source=str(_read_field(raw_anchor, "source", default="anchor_mining")),
        metadata=_json_safe(_read_field(raw_anchor, "metadata", default={})),
    )


def _frame_index_from_id(frame_id: Any, frames: list[_FrameItem]) -> int:
    """Infer a frame index from a frame id."""

    if frame_id is not None:
        frame_id_text = str(frame_id)
        for item in frames:
            if item.frame_id == frame_id_text:
                return item.original_index
        digits = "".join(char for char in frame_id_text if char.isdigit())
        if digits:
            return int(digits)
    return 0


def _score_lookup(scores: Mapping[Any, float] | None, frame_id: Any, frame_index: int, default: float | None) -> float | None:
    """Look up a frame reliability score from common key forms."""

    if scores is None:
        return default
    for key in (frame_id, str(frame_id), frame_index, str(frame_index)):
        if key in scores:
            return float(scores[key])
    return default


def _run_direction(
    backend: Any,
    frames: list[_FrameItem],
    anchor: PropagationAnchor,
    direction: str,
    output_dir: Path,
    config: MultiAnchorPropagationConfig,
) -> AnchorPropagationResult:
    """Run one direction, using reversed-frame fallback when needed."""

    if direction == "forward":
        raw = _call_backend(backend, frames, anchor, "forward", output_dir / "backend" / anchor.anchor_id / "forward")
        result = _normalize_backend_output(raw, anchor, "forward", frames)
        return _filter_result(result, min_index=anchor.frame_index, max_index=len(frames) - 1)
    try:
        raw = _call_backend(backend, frames, anchor, "backward", output_dir / "backend" / anchor.anchor_id / "backward")
        result = _normalize_backend_output(raw, anchor, "backward", frames)
        return _filter_result(result, min_index=0, max_index=anchor.frame_index)
    except Exception as exc:
        if not _is_reverse_unsupported_error(exc):
            raise
        reversed_frames = _create_reversed_frames(frames, output_dir / "cache" / "reversed_frames", config)
        reversed_anchor = PropagationAnchor(
            anchor_id=anchor.anchor_id,
            frame_id=anchor.frame_id,
            frame_index=len(frames) - 1 - anchor.frame_index,
            object_id=anchor.object_id,
            mask=anchor.mask,
            logit=anchor.logit,
            reliability=anchor.reliability,
            source=anchor.source,
            metadata={**anchor.metadata, "reverse_fallback": True},
        )
        raw = _call_backend(
            backend,
            reversed_frames,
            reversed_anchor,
            "forward",
            output_dir / "backend" / anchor.anchor_id / "backward_reversed",
        )
        result = _normalize_backend_output(raw, anchor, "backward", reversed_frames)
        result.warnings.append(f"reverse fallback used after backend error: {type(exc).__name__}: {exc}")
        return _filter_result(result, min_index=0, max_index=anchor.frame_index)


def _call_backend(
    backend: Any,
    frames: list[_FrameItem],
    anchor: PropagationAnchor,
    direction: str,
    output_dir: Path,
) -> Any:
    """Call a duck-typed propagation backend."""

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "frames": [item.path if item.path is not None else item.frame for item in frames],
        "frame_items": frames,
        "anchor": anchor,
        "anchor_frame_index": anchor.frame_index,
        "anchor_frame_id": anchor.frame_id,
        "anchor_mask": anchor.mask,
        "anchor_logit": anchor.logit,
        "object_id": anchor.object_id,
        "direction": direction,
        "output_dir": output_dir,
    }
    if hasattr(backend, "propagate_from_anchor"):
        return _call_with_supported_kwargs(backend.propagate_from_anchor, kwargs)
    if hasattr(backend, "run"):
        return _call_with_supported_kwargs(backend.run, kwargs)
    if callable(backend):
        return _call_with_supported_kwargs(backend, kwargs)
    raise RuntimeError(
        "tracker_backend has no supported propagation API. "
        "Expected propagate_from_anchor(...), run(...), or a callable backend."
    )


def _call_with_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> Any:
    """Call a function using only supported keyword arguments when possible."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return func(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(**filtered)


def _is_reverse_unsupported_error(exc: Exception) -> bool:
    """Return whether an exception means reverse propagation is unsupported."""

    text = str(exc).lower()
    return isinstance(exc, NotImplementedError) or any(token in text for token in ("reverse", "backward", "unsupported", "not supported"))


def _create_reversed_frames(
    frames: list[_FrameItem],
    output_dir: Path,
    config: MultiAnchorPropagationConfig,
) -> list[_FrameItem]:
    """Create a reversed frame directory and return reversed frame metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reversed_items: list[_FrameItem] = []
    for rev_index, original in enumerate(reversed(frames)):
        target = output_dir / f"{rev_index:05d}.jpg"
        if original.path is not None:
            if config.copy_reversed_frames:
                shutil.copyfile(original.path, target)
            else:
                if target.exists():
                    target.unlink()
                target.symlink_to(Path(original.path).resolve())
        else:
            _load_image(original.frame).save(target, quality=95)
        reversed_items.append(
            _FrameItem(
                frame_id=original.frame_id,
                frame_index=rev_index,
                original_index=original.original_index,
                frame=str(target),
                path=str(target),
            )
        )
    return reversed_items


def _normalize_backend_output(
    raw: Any,
    anchor: PropagationAnchor,
    direction: str,
    frames: list[_FrameItem],
) -> AnchorPropagationResult:
    """Normalize backend output to ``AnchorPropagationResult``."""

    if isinstance(raw, AnchorPropagationResult):
        return raw
    masks_payload: Any = None
    logits_payload: Any = None
    warnings: list[str] = []
    if isinstance(raw, dict):
        masks_payload = raw.get("masks", raw.get("mask_paths", raw.get("mask_by_frame")))
        logits_payload = raw.get("logits", raw.get("logit_paths", raw.get("logits_by_frame")))
        warnings = list(raw.get("warnings", []))
    elif isinstance(raw, tuple) and len(raw) >= 2:
        masks_payload, logits_payload = raw[0], raw[1]
    else:
        masks_payload = raw
    masks, mapping = _normalize_output_mapping(masks_payload, frames)
    logits, logit_mapping = _normalize_output_mapping(logits_payload, frames)
    mapping.update(logit_mapping)
    if not masks:
        raise RuntimeError(f"Backend returned no masks for anchor={anchor.anchor_id}, direction={direction}.")
    return AnchorPropagationResult(
        anchor=anchor,
        direction=direction,
        masks=masks,
        logits=logits,
        frame_index_mapping=mapping,
        warnings=warnings,
    )


def _normalize_output_mapping(payload: Any, frames: list[_FrameItem]) -> tuple[dict[str, Any], dict[str, int]]:
    """Normalize list/dict outputs keyed by frame id or index."""

    output: dict[str, Any] = {}
    mapping: dict[str, int] = {}
    if payload is None:
        return output, mapping
    if isinstance(payload, Mapping):
        iterable = payload.items()
    else:
        iterable = enumerate(list(payload))
    for key, value in iterable:
        frame_id, original_index = _frame_key_to_id_index(key, frames)
        output[frame_id] = value
        mapping[frame_id] = original_index
    return output, mapping


def _frame_key_to_id_index(key: Any, frames: list[_FrameItem]) -> tuple[str, int]:
    """Map a backend output key to original frame id and index."""

    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        idx = int(key)
        if 0 <= idx < len(frames):
            return frames[idx].frame_id, frames[idx].original_index
        return str(key), idx
    key_text = str(key)
    for item in frames:
        if item.frame_id == key_text or (item.path is not None and Path(item.path).stem == key_text):
            return item.frame_id, item.original_index
    digits = "".join(char for char in key_text if char.isdigit())
    return key_text, int(digits) if digits else 0


def _filter_result(result: AnchorPropagationResult, min_index: int, max_index: int) -> AnchorPropagationResult:
    """Keep only propagated frames inside an original-frame index range."""

    keep = {
        str(frame_id)
        for frame_id, frame_index in result.frame_index_mapping.items()
        if min_index <= int(frame_index) <= max_index
    }
    result.masks = {frame_id: value for frame_id, value in result.masks.items() if str(frame_id) in keep}
    result.logits = {frame_id: value for frame_id, value in result.logits.items() if str(frame_id) in keep}
    result.frame_index_mapping = {
        frame_id: index for frame_id, index in result.frame_index_mapping.items() if str(frame_id) in keep
    }
    return result


def _fuse_all_frames(
    frames: list[_FrameItem],
    results: list[AnchorPropagationResult],
    config: MultiAnchorPropagationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Fuse per-anchor propagation outputs for every frame."""

    fused_masks: dict[str, np.ndarray] = {}
    debug: dict[str, dict[str, Any]] = {}
    for frame in frames:
        frame_id = frame.frame_id
        all_contrib = _collect_contributions(frame, results, direction=None, config=config)
        forward_contrib = _collect_contributions(frame, results, direction="forward", config=config)
        backward_contrib = _collect_contributions(frame, results, direction="backward", config=config)
        all_mask, all_score, all_mode = _fuse_contributions(all_contrib, config)
        forward_mask, forward_score, _ = _fuse_contributions(forward_contrib, config)
        backward_mask, backward_score, _ = _fuse_contributions(backward_contrib, config)
        selected = "all"
        memory_tree_recovery = False
        final_mask = all_mask
        conflict_iou: float | None = None
        if forward_mask is not None and backward_mask is not None:
            conflict_iou = mask_iou(forward_mask, backward_mask)
            if conflict_iou < config.conflict_iou_threshold:
                if forward_score >= backward_score:
                    final_mask = forward_mask
                    selected = "forward"
                else:
                    final_mask = backward_mask
                    selected = "backward"
                if forward_score < config.low_reliability_threshold and backward_score < config.low_reliability_threshold:
                    memory_tree_recovery = True
        if final_mask is None:
            final_mask = np.zeros((1, 1), dtype=bool)
        fused_masks[frame_id] = final_mask.astype(bool)
        debug[frame_id] = {
            "frame_id": frame_id,
            "frame_index": frame.original_index,
            "fusion_mode": all_mode,
            "selected": selected,
            "forward_score": float(forward_score),
            "backward_score": float(backward_score),
            "all_score": float(all_score),
            "conflict_iou": conflict_iou,
            "memory_tree_recovery_requested": memory_tree_recovery,
            "contributors": [_contribution_debug(item) for item in all_contrib],
        }
    return fused_masks, debug


def _collect_contributions(
    frame: _FrameItem,
    results: list[AnchorPropagationResult],
    direction: str | None,
    config: MultiAnchorPropagationConfig,
) -> list[dict[str, Any]]:
    """Collect weighted contributions for one frame."""

    contributions: list[dict[str, Any]] = []
    for result in results:
        if direction is not None and result.direction != direction:
            continue
        if frame.frame_id not in result.masks and str(frame.original_index) not in result.masks:
            continue
        key = frame.frame_id if frame.frame_id in result.masks else str(frame.original_index)
        weight = result.anchor.reliability * math.exp(-abs(frame.original_index - result.anchor.frame_index) / max(config.tau, config.eps))
        contributions.append(
            {
                "anchor": result.anchor,
                "direction": result.direction,
                "frame_id": frame.frame_id,
                "frame_index": frame.original_index,
                "mask": result.masks.get(key),
                "logit": result.logits.get(key),
                "weight": float(weight),
            }
        )
    return contributions


def _fuse_contributions(
    contributions: list[dict[str, Any]],
    config: MultiAnchorPropagationConfig,
) -> tuple[np.ndarray | None, float, str]:
    """Fuse one contribution set using logits or signed distance maps."""

    if not contributions:
        return None, 0.0, "none"
    has_logit = any(item.get("logit") is not None for item in contributions)
    reference = _reference_array(contributions, prefer_logit=has_logit)
    if reference is None:
        return None, 0.0, "none"
    accum = np.zeros(reference.shape, dtype=np.float32)
    weight_sum = 0.0
    for item in contributions:
        weight = float(item["weight"])
        if has_logit and item.get("logit") is not None:
            field = _load_logit(item["logit"])
            field = _resize_float(field, reference.shape)
        else:
            mask = _load_mask(item.get("mask"))
            field = _signed_distance(mask)
            field = _resize_float(field, reference.shape)
        accum += weight * field
        weight_sum += weight
    fused_field = accum / (weight_sum + config.eps)
    threshold = config.theta if has_logit else 0.0
    return fused_field > threshold, float(weight_sum), "logit" if has_logit else "signed_distance"


def _reference_array(contributions: list[dict[str, Any]], prefer_logit: bool) -> np.ndarray | None:
    """Find a reference array shape for fusion."""

    for item in contributions:
        if prefer_logit and item.get("logit") is not None:
            return _load_logit(item["logit"]).astype(np.float32)
    for item in contributions:
        mask = _load_mask(item.get("mask"))
        if mask.size > 0:
            return mask.astype(np.float32)
    return None


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask to a signed distance field."""

    if mask.size == 0:
        return mask.astype(np.float32)
    binary = mask.astype(np.uint8)
    if binary.max() == 0:
        return -cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5).astype(np.float32)
    if binary.min() == 1:
        return cv2.distanceTransform(binary, cv2.DIST_L2, 5).astype(np.float32)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return np.where(binary > 0, inside, -outside).astype(np.float32)


def _write_outputs(
    result: MultiAnchorPropagationResult,
    frames: list[_FrameItem],
    output_dir: Path,
    config: MultiAnchorPropagationConfig,
) -> None:
    """Write masks, logits, fusion debug, and side-by-side debug video."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fused_dir = output_dir / "fused_masks"
    per_anchor_dir = output_dir / "per_anchor"
    per_anchor_logits_dir = output_dir / "per_anchor_logits"
    fused_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, mask in result.fused_masks.items():
        _save_mask(mask, fused_dir / f"{frame_id}.png")
    for propagation in result.per_anchor_results:
        anchor_dir = per_anchor_dir / _safe_name(propagation.anchor.anchor_id) / propagation.direction
        for frame_id, mask in propagation.masks.items():
            _save_mask(_load_mask(mask), anchor_dir / f"{frame_id}.png")
        for frame_id, logit in propagation.logits.items():
            logit_dir = per_anchor_logits_dir / _safe_name(propagation.anchor.anchor_id) / propagation.direction
            logit_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(logit_dir / f"{frame_id}.npz", logit=_load_logit(logit))
    debug_path = output_dir / "fusion_debug.json"
    debug_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    video_path = output_dir / "multi_anchor_debug.mp4"
    if _write_debug_video(frames, result, video_path, config):
        result.output_paths["debug_video"] = str(video_path)
    else:
        result.warnings.append("multi_anchor_debug.mp4 skipped because frames could not be read.")
    result.output_paths.update(
        {
            "fused_masks": str(fused_dir),
            "per_anchor": str(per_anchor_dir),
            "per_anchor_logits": str(per_anchor_logits_dir),
            "fusion_debug": str(debug_path),
        }
    )
    debug_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def _write_debug_video(
    frames: list[_FrameItem],
    result: MultiAnchorPropagationResult,
    output_path: Path,
    config: MultiAnchorPropagationConfig,
) -> bool:
    """Write original/fused/forward/backward side-by-side debug video."""

    panels: list[np.ndarray] = []
    for frame in frames:
        image = _try_load_image(frame.path or frame.frame)
        if image is None:
            return False
        fused = result.fused_masks.get(frame.frame_id)
        forward = _direction_mask_for_frame(frame, result, "forward")
        backward = _direction_mask_for_frame(frame, result, "backward")
        panel = np.concatenate(
            [
                np.asarray(image),
                _overlay(image, fused, "fused"),
                _overlay(image, forward, "forward"),
                _overlay(image, backward, "backward"),
            ],
            axis=1,
        )
        panels.append(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    if not panels:
        return False
    height, width = panels[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config.debug_video_fps),
        (width, height),
    )
    if not writer.isOpened():
        return False
    for panel in panels:
        if panel.shape[:2] != (height, width):
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(panel)
    writer.release()
    return True


def _direction_mask_for_frame(
    frame: _FrameItem,
    result: MultiAnchorPropagationResult,
    direction: str,
) -> np.ndarray | None:
    """Fuse one direction for debug-video display using saved contributors."""

    masks = [
        _load_mask(prop.masks[frame.frame_id])
        for prop in result.per_anchor_results
        if prop.direction == direction and frame.frame_id in prop.masks
    ]
    if not masks:
        return None
    accum = np.zeros_like(masks[0], dtype=np.float32)
    for mask in masks:
        accum += _resize_float(mask.astype(np.float32), masks[0].shape)
    return accum > 0


def _overlay(image: Image.Image, mask: np.ndarray | None, label: str) -> np.ndarray:
    """Draw a translucent mask overlay and label."""

    rgb = np.asarray(image.convert("RGB")).copy()
    if mask is not None:
        mask = _resize_mask(mask, (image.height, image.width))
        color = np.asarray([255, 64, 64], dtype=np.uint8)
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, 150, 22], fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return np.asarray(pil)


def _load_mask(mask: MaskLike) -> np.ndarray:
    """Load mask-like input as boolean array."""

    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    if isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        return np.any(array[..., :3] > 0, axis=-1)
    return array > 0


def _load_logit(logit: LogitLike) -> np.ndarray:
    """Load logit-like input as float32 array."""

    if logit is None:
        raise ValueError("logit is None")
    if isinstance(logit, np.ndarray):
        array = logit
    else:
        path = Path(logit)
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as payload:
                key = "logit" if "logit" in payload.files else sorted(payload.files)[0]
                array = payload[key]
        else:
            array = np.asarray(Image.open(path), dtype=np.float32)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    return array.astype(np.float32)


def _load_image(frame: Any) -> Image.Image:
    """Load image-like input as RGB PIL image."""

    if isinstance(frame, (str, Path)):
        return Image.open(frame).convert("RGB")
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return Image.fromarray(array[..., :3].astype(np.uint8)).convert("RGB")


def _try_load_image(frame: Any) -> Image.Image | None:
    """Load an image, returning None on failure."""

    try:
        return _load_image(frame)
    except Exception:
        return None


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to ``(height, width)``."""

    if mask.shape[:2] == shape:
        return mask.astype(bool)
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    ) > 0


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a float field to ``(height, width)``."""

    if array.shape[:2] == shape:
        return array.astype(np.float32)
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    """Save a boolean mask as an 8-bit PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _contribution_debug(item: dict[str, Any]) -> dict[str, Any]:
    """Return debug metadata for one weighted contribution."""

    anchor = item["anchor"]
    return {
        "anchor_id": anchor.anchor_id,
        "anchor_frame_index": anchor.frame_index,
        "anchor_frame_id": anchor.frame_id,
        "direction": item["direction"],
        "weight": float(item["weight"]),
        "reliability": float(anchor.reliability),
        "has_logit": item.get("logit") is not None,
    }


def _mask_ref(mask: MaskLike) -> str | dict[str, Any] | None:
    """Summarize a mask reference without embedding arrays."""

    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return str(mask)
    if isinstance(mask, Image.Image):
        return {"type": "PIL.Image", "size": list(mask.size)}
    array = np.asarray(mask)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _logit_ref(logit: LogitLike) -> str | dict[str, Any] | None:
    """Summarize a logit reference without embedding arrays."""

    if logit is None:
        return None
    if isinstance(logit, (str, Path)):
        return str(logit)
    array = np.asarray(logit)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _safe_name(value: Any) -> str:
    """Return a filesystem-safe name."""

    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _read_field(item: Any, *names: str, default: Any = None) -> Any:
    """Read the first available field from a dictionary or object."""

    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _json_safe(value: Any) -> Any:
    """Convert common values to JSON-safe primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return str(value)


class _PrecomputedPropagationBackend:
    """CLI helper backend backed by precomputed masks/logits in anchors JSON."""

    def __init__(self, anchors: list[dict[str, Any]]) -> None:
        """Store precomputed anchor payloads by anchor id."""

        self.payloads = {str(anchor.get("anchor_id", "")): anchor for anchor in anchors}

    def propagate_from_anchor(
        self,
        anchor: PropagationAnchor,
        direction: str,
        **_: Any,
    ) -> dict[str, Any]:
        """Return precomputed propagation for one anchor and direction."""

        payload = self.payloads.get(anchor.anchor_id)
        if payload is None:
            raise RuntimeError(f"No precomputed payload for anchor_id={anchor.anchor_id}")
        direction_payload = payload.get("propagations", {}).get(direction)
        if direction_payload is None:
            direction_payload = payload.get(direction)
        if direction_payload is None:
            raise RuntimeError(
                "CLI requires precomputed propagation masks/logits in anchors-json; "
                f"missing direction={direction} for anchor_id={anchor.anchor_id}."
            )
        return direction_payload


def _load_anchors_json(path: str | Path) -> list[dict[str, Any]]:
    """Load anchors JSON for CLI replay."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "anchors" in payload:
        payload = payload["anchors"]
    if not isinstance(payload, list):
        raise ValueError("anchors JSON must be a list or contain an 'anchors' list")
    return [dict(anchor) for anchor in payload]


def _frame_paths_from_dir(frames_dir: str | Path) -> list[str]:
    """Return naturally sorted image frames from a directory."""

    root = Path(frames_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [str(path) for path in sorted((p for p in root.iterdir() if p.suffix.lower() in suffixes), key=_natural_key)]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run multi-anchor bidirectional propagation fusion.")
    parser.add_argument("--frames-dir", required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--first-mask", required=True, help="First-frame object mask path.")
    parser.add_argument("--anchors-json", required=True, help="Anchor JSON with masks and optional precomputed propagations.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--object-id", default="1", help="Object id to process.")
    parser.add_argument("--tau", type=float, default=25.0)
    parser.add_argument("--theta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    """Run CLI replay using precomputed propagation entries."""

    args = _parse_args()
    anchors = _load_anchors_json(args.anchors_json)
    first_anchor_id = f"{args.object_id}_first"
    non_first_anchors = [anchor for anchor in anchors if str(anchor.get("anchor_id", "")) != first_anchor_id]
    result = run_multi_anchor_bidirectional_propagation(
        first_frame_mask=args.first_mask,
        anchor_masks=non_first_anchors,
        video_frames=_frame_paths_from_dir(args.frames_dir),
        tracker_backend=_PrecomputedPropagationBackend(anchors),
        reliability_scores=None,
        output_dir=args.output_dir,
        object_id=args.object_id,
        config=MultiAnchorPropagationConfig(tau=args.tau, theta=args.theta),
    )
    print(json.dumps({"output_dir": args.output_dir, "num_frames": len(result.fused_masks), "warnings": result.warnings}, indent=2))


if __name__ == "__main__":
    main()

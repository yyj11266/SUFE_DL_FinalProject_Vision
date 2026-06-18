"""TTA fusion, object-mask postprocess, and multi-object merge utilities."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from src.vos.reliability import bbox_iou, mask_area, mask_to_bbox


MaskLike = np.ndarray | Image.Image | str | Path | None
LogitLike = np.ndarray | str | Path | None
BBox = list[float] | tuple[float, float, float, float] | np.ndarray | None


@dataclass(slots=True)
class TTAConfig:
    """Configuration for deterministic test-time augmentation transforms."""

    variants: list[str] = field(default_factory=lambda: ["original", "hflip", "resize", "box_jitter"])
    resize_long_sides: list[int] = field(default_factory=lambda: [1024, 1536])
    box_jitter_ratios: list[float] = field(default_factory=lambda: [-0.05, 0.0, 0.05])
    min_gpu_memory_gb_for_1536: float = 14.0
    logit_threshold: float = 0.0


@dataclass(slots=True)
class TTAVariant:
    """One TTA transform specification."""

    name: str
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    hflip: bool = False
    bbox: list[float] | None = None
    bbox_jitter_ratio: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe variant metadata."""

        return asdict(self)


@dataclass(slots=True)
class TTAFusionResult:
    """Output of TTA result fusion."""

    mask: np.ndarray
    logit: np.ndarray
    used_logits: bool
    num_results: int
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostprocessConfig:
    """Configuration for target-aware mask postprocessing."""

    tiny_min_area_abs: int = 4
    tiny_min_area_ratio: float = 0.05
    regular_min_area_abs: int = 16
    regular_min_area_ratio: float = 0.01
    close_kernel_size: int = 3
    open_kernel_size: int = 3
    smooth_kernel_size: int = 3
    keep_largest_component: bool = True
    fragmented_largest_ratio_threshold: float = 0.65
    fragmented_component_count_threshold: int = 3
    area_min_recent_ratio: float = 0.15
    area_max_recent_ratio: float = 6.0


@dataclass(slots=True)
class MergeConfig:
    """Configuration for multi-object mask/logit merge."""

    gamma: float = 0.20
    foreground_threshold: float = 0.0
    mask_encoding: str = "indexed_png"
    binary_foreground_value: int = 255
    dtype: str = "uint8"


@dataclass(slots=True)
class ObjectMaskPrediction:
    """One object's mask/logit prediction and merge metadata."""

    object_id: int
    mask: MaskLike = None
    logit: LogitLike = None
    reliability: float = 0.0
    bbox: list[float] | None = None
    previous_bbox: list[float] | None = None
    area_initial: float | None = None
    target_type: str = "regular"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe prediction summary."""

        return {
            "object_id": int(self.object_id),
            "mask": _mask_ref(self.mask),
            "logit": _logit_ref(self.logit),
            "reliability": float(self.reliability),
            "bbox": _json_box(self.bbox),
            "previous_bbox": _json_box(self.previous_bbox),
            "area_initial": self.area_initial,
            "target_type": self.target_type,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class PostprocessResult:
    """Postprocess output for one object mask."""

    mask: np.ndarray
    abnormal_area: bool
    area: int
    area_min: float | None
    area_max: float | None
    min_area: int
    fragmented: bool
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MergeResult:
    """Multi-object merge output."""

    indexed_mask: np.ndarray
    debug: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def build_tta_variants(
    image_size: tuple[int, int] | list[int],
    bbox: BBox = None,
    gpu_memory_gb: float | None = None,
    config: TTAConfig | dict[str, Any] | None = None,
) -> list[TTAVariant]:
    """Build deterministic TTA variants for an image size ``(height, width)``."""

    cfg = _as_tta_config(config)
    height, width = _size_hw(image_size)
    variants: list[TTAVariant] = []
    if "original" in cfg.variants:
        variants.append(TTAVariant("original", (height, width), (height, width), bbox=_json_box(bbox)))
    if "hflip" in cfg.variants:
        variants.append(TTAVariant("hflip", (height, width), (height, width), hflip=True, bbox=_flip_box(_json_box(bbox), width)))
    if "resize" in cfg.variants:
        for long_side in cfg.resize_long_sides:
            if int(long_side) == 1536 and (gpu_memory_gb is None or gpu_memory_gb < cfg.min_gpu_memory_gb_for_1536):
                continue
            output_size = _resize_long_side((height, width), int(long_side))
            variants.append(
                TTAVariant(
                    name=f"resize_{int(long_side)}",
                    original_size=(height, width),
                    output_size=output_size,
                    bbox=_scale_box(_json_box(bbox), (height, width), output_size),
                    metadata={"long_side": int(long_side)},
                )
            )
    if "box_jitter" in cfg.variants and bbox is not None:
        for ratio in cfg.box_jitter_ratios:
            variants.append(
                TTAVariant(
                    name=f"box_jitter_{ratio:+.2f}",
                    original_size=(height, width),
                    output_size=(height, width),
                    bbox=_jitter_box(_json_box(bbox), (height, width), float(ratio)),
                    bbox_jitter_ratio=float(ratio),
                )
            )
    return variants


def apply_tta_transform(
    array: np.ndarray | Image.Image,
    variant: TTAVariant,
    is_mask: bool = False,
) -> np.ndarray:
    """Apply a TTA transform to an image, mask, or logit array."""

    data = np.asarray(array)
    if variant.hflip:
        data = np.flip(data, axis=1)
    if data.shape[:2] != variant.output_size:
        interpolation = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        data = _resize_array(data, variant.output_size, interpolation)
    return data


def invert_tta_result(
    mask: MaskLike = None,
    logit: LogitLike = None,
    variant: TTAVariant | dict[str, Any] | None = None,
    original_size: tuple[int, int] | list[int] | None = None,
) -> dict[str, np.ndarray | None]:
    """Invert a TTA mask/logit result back to original image coordinates."""

    if variant is None:
        if original_size is None:
            raise ValueError("Either variant or original_size must be provided.")
        variant_obj = TTAVariant("identity", _size_hw(original_size), _size_hw(original_size))
    elif isinstance(variant, TTAVariant):
        variant_obj = variant
    else:
        variant_obj = TTAVariant(**variant)
    target_size = _size_hw(original_size) if original_size is not None else variant_obj.original_size
    out_mask: np.ndarray | None = None
    out_logit: np.ndarray | None = None
    if mask is not None:
        out_mask = _load_mask(mask)
        if out_mask.shape[:2] != target_size:
            out_mask = _resize_array(out_mask.astype(np.uint8), target_size, Image.Resampling.NEAREST) > 0
        if variant_obj.hflip:
            out_mask = np.flip(out_mask, axis=1)
    if logit is not None:
        out_logit = _load_logit(logit)
        if out_logit.shape[:2] != target_size:
            out_logit = _resize_float(out_logit, target_size)
        if variant_obj.hflip:
            out_logit = np.flip(out_logit, axis=1)
    return {"mask": out_mask, "logit": out_logit}


def fuse_tta_results(
    results: Iterable[Any],
    config: TTAConfig | dict[str, Any] | None = None,
) -> TTAFusionResult:
    """Fuse inverse-transformed TTA results by averaging logits or masks."""

    cfg = _as_tta_config(config)
    items = list(results)
    if not items:
        raise ValueError("No TTA results to fuse.")
    logits = [_load_logit(_read_field(item, "logit", default=None)) for item in items if _read_field(item, "logit", default=None) is not None]
    if logits:
        shape = logits[0].shape[:2]
        logit_planes = [_resize_float(logit, shape) for logit in logits]
        mask_fallback_count = 0
        for item in items:
            if _read_field(item, "logit", default=None) is not None:
                continue
            mask_value = _read_field(item, "mask", default=None)
            if mask_value is None:
                continue
            logit_planes.append(_resize_mask(_load_mask(mask_value), shape).astype(np.float32) - 0.5)
            mask_fallback_count += 1
        fused_logit = np.mean(logit_planes, axis=0).astype(np.float32)
        return TTAFusionResult(
            mask=fused_logit > cfg.logit_threshold,
            logit=fused_logit,
            used_logits=True,
            num_results=len(logit_planes),
            debug={"mode": "logits_mean", "mask_fallback_count": mask_fallback_count},
        )
    masks = [_load_mask(_read_field(item, "mask", default=item)) for item in items]
    shape = next((mask.shape[:2] for mask in masks if mask.size > 0), None)
    if shape is None:
        raise ValueError("TTA masks are all empty.")
    fused_logit = np.mean([_resize_mask(mask, shape).astype(np.float32) for mask in masks], axis=0).astype(np.float32) - 0.5
    return TTAFusionResult(
        mask=fused_logit > 0.0,
        logit=fused_logit,
        used_logits=False,
        num_results=len(masks),
        debug={"mode": "mask_mean_minus_0.5"},
    )


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than ``min_area``."""

    binary = _load_mask(mask)
    if binary.size == 0:
        return binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    output = np.zeros_like(binary, dtype=bool)
    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            output[labels == label] = True
    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask."""

    binary = _load_mask(mask)
    if binary.size == 0:
        return binary
    flood = binary.astype(np.uint8).copy()
    padded = np.pad(flood, 1, mode="constant", constant_values=0)
    cv2.floodFill(padded, None, (0, 0), 1)
    exterior = padded[1:-1, 1:-1] > 0
    holes = ~exterior & ~binary
    return binary | holes


def morphological_close_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply morphological close followed by open."""

    binary = _load_mask(mask)
    if binary.size == 0:
        return binary
    closed = _morphological_operation(binary, cv2.MORPH_CLOSE, kernel_size)
    return _morphological_operation(closed, cv2.MORPH_OPEN, kernel_size)


def smooth_boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth mask boundaries with a small blur and threshold."""

    binary = _load_mask(mask)
    if binary.size == 0:
        return binary
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return binary
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (kernel_size, kernel_size), 0)
    return blurred >= 0.5


def _morphological_operation(mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    """Apply one OpenCV morphology operation with an elliptical kernel."""

    binary = _load_mask(mask)
    if binary.size == 0:
        return binary
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary.astype(np.uint8), operation, kernel) > 0


def postprocess_object_mask(
    prediction: ObjectMaskPrediction | dict[str, Any],
    recent_areas: Sequence[float] | None = None,
    config: PostprocessConfig | dict[str, Any] | None = None,
) -> PostprocessResult:
    """Postprocess one object mask with target-aware area rules."""

    cfg = _as_post_config(config)
    pred = _prediction_from_any(prediction)
    if pred.mask is not None:
        mask = _load_mask(pred.mask)
    elif pred.logit is not None:
        mask = _load_logit(pred.logit) > 0
    else:
        raise ValueError("Object prediction must contain either mask or logit for postprocess.")
    area_initial = float(pred.area_initial if pred.area_initial is not None else max(1, int(mask.sum())))
    target_type = str(pred.target_type).lower()
    if target_type == "tiny":
        min_area = int(max(cfg.tiny_min_area_abs, round(cfg.tiny_min_area_ratio * area_initial)))
    else:
        min_area = int(max(cfg.regular_min_area_abs, round(cfg.regular_min_area_ratio * area_initial)))
    before_area = int(mask.sum())
    mask = remove_small_components(mask, min_area)
    mask = fill_holes(mask)
    mask = _morphological_operation(mask, cv2.MORPH_CLOSE, cfg.close_kernel_size)
    mask = _morphological_operation(mask, cv2.MORPH_OPEN, cfg.open_kernel_size)
    fragmented, component_debug = _is_fragmented(mask, cfg)
    if cfg.keep_largest_component and not fragmented:
        mask = _keep_largest_component(mask)
    mask = smooth_boundary(mask, cfg.smooth_kernel_size)
    area = int(mask.sum())
    area_min: float | None = None
    area_max: float | None = None
    abnormal = False
    if recent_areas:
        median_area = float(np.median(np.asarray(recent_areas, dtype=np.float32)))
        area_min = cfg.area_min_recent_ratio * median_area
        area_max = cfg.area_max_recent_ratio * median_area
        abnormal = bool(area < area_min or area > area_max)
    return PostprocessResult(
        mask=mask.astype(bool),
        abnormal_area=abnormal,
        area=area,
        area_min=area_min,
        area_max=area_max,
        min_area=min_area,
        fragmented=fragmented,
        debug={
            "before_area": before_area,
            "target_type": pred.target_type,
            "area_initial": area_initial,
            "component_debug": component_debug,
        },
    )


def merge_object_predictions(
    predictions: Iterable[ObjectMaskPrediction | dict[str, Any]],
    previous_indexed_mask: np.ndarray | Image.Image | str | Path | None = None,
    format_spec: Any = None,
    config: MergeConfig | dict[str, Any] | None = None,
) -> MergeResult:
    """Merge object-level predictions into one indexed or binary mask."""

    cfg = _as_merge_config(config)
    cfg = _merge_config_with_format(cfg, format_spec)
    preds = [_prediction_from_any(pred) for pred in predictions]
    if not preds:
        raise ValueError("No object predictions to merge.")
    prev = _load_indexed_mask(previous_indexed_mask) if previous_indexed_mask is not None else None
    shape = _prediction_shape(preds)
    warnings: list[str] = []
    if any(pred.logit is not None for pred in preds):
        indexed, debug = _merge_logits(preds, shape, cfg)
    else:
        indexed, debug = _merge_binary(preds, shape, prev, cfg)
    if cfg.mask_encoding == "binary_png":
        indexed = np.where(indexed > 0, cfg.binary_foreground_value, 0).astype(np.uint8)
    else:
        indexed = indexed.astype(np.uint8)
    return MergeResult(indexed_mask=indexed, debug=debug, warnings=warnings)


def save_indexed_png(
    indexed_mask: np.ndarray,
    output_path: str | Path,
    format_spec: Any = None,
    palette: Sequence[int] | None = None,
) -> None:
    """Save an indexed or binary PNG without changing object-id pixel values."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = _format_spec_field(format_spec, "mask_encoding", "indexed_png")
    array = np.asarray(indexed_mask)
    if encoding == "binary_png":
        array = np.where(array > 0, 255, 0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    if palette is not None:
        image = image.convert("P")
        image.putpalette(list(palette))
    image.save(path)


def _merge_logits(preds: list[ObjectMaskPrediction], shape: tuple[int, int], config: MergeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge predictions using logit scores plus reliability bias."""

    score_planes: list[np.ndarray] = []
    object_ids: list[int] = []
    for pred in preds:
        if pred.logit is not None:
            logit = _resize_float(_load_logit(pred.logit), shape)
        else:
            logit = _resize_mask(_load_mask(pred.mask), shape).astype(np.float32) - 0.5
        score_planes.append(logit + config.gamma * float(pred.reliability))
        object_ids.append(int(pred.object_id))
    scores = np.stack(score_planes, axis=0)
    winners = np.argmax(scores, axis=0)
    max_scores = np.max(scores, axis=0)
    indexed = np.zeros(shape, dtype=np.uint8)
    foreground = max_scores > config.foreground_threshold
    for idx, object_id in enumerate(object_ids):
        indexed[foreground & (winners == idx)] = np.uint8(min(max(object_id, 0), 255))
    return indexed, {"mode": "logits", "object_ids": object_ids, "gamma": config.gamma}


def _merge_binary(
    preds: list[ObjectMaskPrediction],
    shape: tuple[int, int],
    previous_indexed_mask: np.ndarray | None,
    config: MergeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge binary masks with reliability and temporal tie-breaks."""

    indexed = np.zeros(shape, dtype=np.uint8)
    owner_score: dict[int, dict[str, float]] = {}
    for pred in sorted(preds, key=lambda item: float(item.reliability), reverse=True):
        object_id = int(pred.object_id)
        mask = _resize_mask(_load_mask(pred.mask), shape)
        for y, x in np.argwhere(mask):
            current = int(indexed[y, x])
            if current == 0:
                indexed[y, x] = np.uint8(min(max(object_id, 0), 255))
                continue
            winner = _binary_overlap_winner(current, pred, preds, previous_indexed_mask, shape)
            indexed[y, x] = np.uint8(min(max(winner, 0), 255))
        owner_score[object_id] = {
            "reliability": float(pred.reliability),
            "area": float(mask.sum()),
            "area_initial": float(pred.area_initial if pred.area_initial is not None else max(1, mask.sum())),
        }
    return indexed, {"mode": "binary", "object_scores": owner_score, "gamma": config.gamma}


def _binary_overlap_winner(
    current_object_id: int,
    new_pred: ObjectMaskPrediction,
    preds: list[ObjectMaskPrediction],
    previous_indexed_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> int:
    """Choose owner for a binary-mask overlap pixel."""

    current_pred = next((pred for pred in preds if int(pred.object_id) == int(current_object_id)), None)
    if current_pred is None:
        return int(new_pred.object_id)
    if float(new_pred.reliability) != float(current_pred.reliability):
        return int(new_pred.object_id if new_pred.reliability > current_pred.reliability else current_object_id)
    new_iou = _previous_bbox_iou(new_pred, previous_indexed_mask)
    cur_iou = _previous_bbox_iou(current_pred, previous_indexed_mask)
    if new_iou != cur_iou:
        return int(new_pred.object_id if new_iou > cur_iou else current_object_id)
    new_delta = _area_change_abs(new_pred, shape)
    cur_delta = _area_change_abs(current_pred, shape)
    if new_delta != cur_delta:
        return int(new_pred.object_id if new_delta < cur_delta else current_object_id)
    return int(min(current_object_id, int(new_pred.object_id)))


def _previous_bbox_iou(pred: ObjectMaskPrediction, previous_indexed_mask: np.ndarray | None) -> float:
    """Return IoU to previous-frame bbox for tie-breaking."""

    current_box = _json_box(pred.bbox if pred.bbox is not None else mask_to_bbox(_load_mask(pred.mask)))
    previous_box = _json_box(pred.previous_bbox)
    if previous_box is None and previous_indexed_mask is not None:
        previous_box = mask_to_bbox(previous_indexed_mask == int(pred.object_id))
    return bbox_iou(current_box, previous_box)


def _area_change_abs(pred: ObjectMaskPrediction, shape: tuple[int, int]) -> float:
    """Return absolute area ratio change against initial area."""

    area = float(_resize_mask(_load_mask(pred.mask), shape).sum())
    initial = float(pred.area_initial if pred.area_initial is not None else max(1.0, area))
    return abs(math.log((area + 1e-6) / (initial + 1e-6)))


def _is_fragmented(mask: np.ndarray, config: PostprocessConfig) -> tuple[bool, dict[str, Any]]:
    """Detect fragmented objects from connected-component statistics."""

    binary = _load_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    total = int(sum(areas))
    if total == 0:
        return False, {"component_count": 0, "largest_ratio": 0.0}
    largest_ratio = max(areas) / float(total)
    component_count = len(areas)
    fragmented = largest_ratio < config.fragmented_largest_ratio_threshold or component_count >= config.fragmented_component_count_threshold
    return fragmented, {"component_count": component_count, "largest_ratio": largest_ratio}


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""

    binary = _load_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return binary
    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def _prediction_from_any(item: ObjectMaskPrediction | dict[str, Any]) -> ObjectMaskPrediction:
    """Normalize prediction-like input to ``ObjectMaskPrediction``."""

    if isinstance(item, ObjectMaskPrediction):
        return item
    return ObjectMaskPrediction(
        object_id=int(item.get("object_id", item.get("id", 1))),
        mask=item.get("mask", item.get("mask_path")),
        logit=item.get("logit", item.get("logit_path")),
        reliability=float(item.get("reliability", item.get("R_t", 0.0))),
        bbox=_json_box(item.get("bbox")),
        previous_bbox=_json_box(item.get("previous_bbox")),
        area_initial=item.get("area_initial"),
        target_type=str(item.get("target_type", "regular")),
        metadata=dict(item.get("metadata", {})),
    )


def _prediction_shape(preds: list[ObjectMaskPrediction]) -> tuple[int, int]:
    """Infer output shape from the first available mask or logit."""

    for pred in preds:
        if pred.logit is not None:
            return _load_logit(pred.logit).shape[:2]
    for pred in preds:
        if pred.mask is not None:
            mask = _load_mask(pred.mask)
            if mask.size > 0:
                return mask.shape[:2]
    raise ValueError("Could not infer prediction shape from masks/logits.")


def _as_tta_config(config: TTAConfig | dict[str, Any] | None) -> TTAConfig:
    """Normalize TTA config input."""

    if config is None:
        return TTAConfig()
    if isinstance(config, TTAConfig):
        return config
    allowed = TTAConfig.__dataclass_fields__.keys()
    return TTAConfig(**{key: value for key, value in config.items() if key in allowed})


def _as_post_config(config: PostprocessConfig | dict[str, Any] | None) -> PostprocessConfig:
    """Normalize postprocess config input."""

    if config is None:
        return PostprocessConfig()
    if isinstance(config, PostprocessConfig):
        return config
    allowed = PostprocessConfig.__dataclass_fields__.keys()
    return PostprocessConfig(**{key: value for key, value in config.items() if key in allowed})


def _as_merge_config(config: MergeConfig | dict[str, Any] | None) -> MergeConfig:
    """Normalize merge config input."""

    if config is None:
        return MergeConfig()
    if isinstance(config, MergeConfig):
        return config
    allowed = MergeConfig.__dataclass_fields__.keys()
    return MergeConfig(**{key: value for key, value in config.items() if key in allowed})


def _merge_config_with_format(config: MergeConfig, format_spec: Any) -> MergeConfig:
    """Return merge config adjusted by an optional FormatSpec-like object."""

    encoding = _format_spec_field(format_spec, "mask_encoding", config.mask_encoding)
    return MergeConfig(
        gamma=config.gamma,
        foreground_threshold=config.foreground_threshold,
        mask_encoding=encoding,
        binary_foreground_value=config.binary_foreground_value,
        dtype=config.dtype,
    )


def _format_spec_field(format_spec: Any, name: str, default: Any) -> Any:
    """Read a field from a FormatSpec-like object or dictionary/path."""

    if format_spec is None:
        return default
    if isinstance(format_spec, (str, Path)):
        payload = json.loads(Path(format_spec).read_text(encoding="utf-8"))
        return payload.get(name, default)
    if isinstance(format_spec, dict):
        return format_spec.get(name, default)
    return getattr(format_spec, name, default)


def _size_hw(size: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize a size to ``(height, width)``."""

    if len(size) != 2:
        raise ValueError(f"Expected size with two values, got {size!r}")
    return int(size[0]), int(size[1])


def _resize_long_side(size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """Resize a ``(height, width)`` pair by long side."""

    height, width = size
    current = max(height, width)
    if current <= 0:
        return height, width
    scale = long_side / float(current)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _flip_box(box: list[float] | None, width: int) -> list[float] | None:
    """Horizontally flip an inclusive bbox."""

    if box is None:
        return None
    return [float(width - 1 - box[2]), float(box[1]), float(width - 1 - box[0]), float(box[3])]


def _scale_box(box: list[float] | None, src_size: tuple[int, int], dst_size: tuple[int, int]) -> list[float] | None:
    """Scale a bbox from source size to destination size."""

    if box is None:
        return None
    src_h, src_w = src_size
    dst_h, dst_w = dst_size
    sx = dst_w / max(1.0, float(src_w))
    sy = dst_h / max(1.0, float(src_h))
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def _jitter_box(box: list[float] | None, size: tuple[int, int], ratio: float) -> list[float] | None:
    """Jitter bbox sides by a ratio of bbox width/height."""

    if box is None:
        return None
    height, width = size
    x0, y0, x1, y1 = box
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    return [
        float(np.clip(x0 - ratio * bw, 0, width - 1)),
        float(np.clip(y0 - ratio * bh, 0, height - 1)),
        float(np.clip(x1 + ratio * bw, 0, width - 1)),
        float(np.clip(y1 + ratio * bh, 0, height - 1)),
    ]


def _json_box(box: BBox) -> list[float] | None:
    """Normalize bbox to a JSON-safe list."""

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


def _read_field(item: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dictionary or object."""

    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _load_mask(mask: Any) -> np.ndarray:
    """Load a mask-like object as a boolean array."""

    if isinstance(mask, bool):
        return np.asarray([[mask]], dtype=bool)
    if isinstance(mask, np.ndarray):
        array = mask
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    elif isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif mask is None:
        return np.zeros((0, 0), dtype=bool)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        return np.any(array[..., :3] > 0, axis=-1)
    return array > 0


def _load_logit(logit: Any) -> np.ndarray:
    """Load a logit-like object as float32 array."""

    if isinstance(logit, np.ndarray):
        array = logit
    elif isinstance(logit, (str, Path)):
        path = Path(logit)
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as payload:
                key = "logit" if "logit" in payload.files else ("logits" if "logits" in payload.files else sorted(payload.files)[0])
                array = payload[key]
        else:
            array = np.asarray(Image.open(path), dtype=np.float32)
    else:
        array = np.asarray(logit)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 2:
        array = np.squeeze(array)
    return array.astype(np.float32)


def _load_indexed_mask(mask: MaskLike) -> np.ndarray:
    """Load an indexed mask without binarizing object ids."""

    if isinstance(mask, np.ndarray):
        array = mask
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    elif isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint8)


def _resize_array(array: np.ndarray, size_hw: tuple[int, int], interpolation: Image.Resampling) -> np.ndarray:
    """Resize a numpy array with Pillow."""

    height, width = size_hw
    if array.shape[:2] == (height, width):
        return array
    image = Image.fromarray(array.astype(np.float32) if array.dtype.kind == "f" else array)
    return np.asarray(image.resize((width, height), interpolation))


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to shape ``(height, width)``."""

    mask = _load_mask(mask)
    if mask.shape[:2] == shape:
        return mask.astype(bool)
    return _resize_array(mask.astype(np.uint8), shape, Image.Resampling.NEAREST) > 0


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a float array to shape ``(height, width)``."""

    if array.shape[:2] == shape:
        return array.astype(np.float32)
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _mask_ref(mask: MaskLike) -> str | dict[str, Any] | None:
    """Summarize mask references for debug JSON."""

    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        return str(mask)
    if isinstance(mask, Image.Image):
        return {"type": "PIL.Image", "size": list(mask.size)}
    array = np.asarray(mask)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _logit_ref(logit: LogitLike) -> str | dict[str, Any] | None:
    """Summarize logit references for debug JSON."""

    if logit is None:
        return None
    if isinstance(logit, (str, Path)):
        return str(logit)
    array = np.asarray(logit)
    return {"type": "ndarray", "shape": list(array.shape), "dtype": str(array.dtype)}


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-safe primitives."""

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


def _load_predictions_json(path: str | Path) -> list[dict[str, Any]]:
    """Load CLI predictions JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "predictions" in payload:
        payload = payload["predictions"]
    if not isinstance(payload, list):
        raise ValueError("predictions JSON must be a list or contain a 'predictions' list")
    return [dict(item) for item in payload]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Postprocess and merge object masks into an indexed PNG.")
    parser.add_argument("--predictions-json", required=True, help="JSON list of object predictions.")
    parser.add_argument("--output-png", required=True, help="Output indexed/binary PNG path.")
    parser.add_argument("--format-spec", default=None, help="Optional FormatSpec JSON path.")
    parser.add_argument("--debug-json", default=None, help="Optional debug JSON output path.")
    parser.add_argument("--previous-indexed-mask", default=None, help="Optional previous indexed mask for overlap tie-breaks.")
    return parser.parse_args()


def main() -> None:
    """Run postprocess CLI."""

    args = _parse_args()
    predictions = [_prediction_from_any(item) for item in _load_predictions_json(args.predictions_json)]
    merge = merge_object_predictions(
        predictions,
        previous_indexed_mask=args.previous_indexed_mask,
        format_spec=args.format_spec,
    )
    save_indexed_png(merge.indexed_mask, args.output_png, format_spec=args.format_spec)
    debug_path = Path(args.debug_json) if args.debug_json else Path(args.output_png).with_suffix(".debug.json")
    debug_path.write_text(json.dumps(merge.debug, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output_png": args.output_png, "debug_json": str(debug_path), "warnings": merge.warnings}, indent=2))


if __name__ == "__main__":
    main()

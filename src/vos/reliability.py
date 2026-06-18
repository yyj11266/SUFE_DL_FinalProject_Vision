"""Reliability, drift, and lost-state scoring for video object segmentation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BBox = list[float] | tuple[float, float, float, float] | np.ndarray | None
FeatureFn = Callable[..., np.ndarray]


@dataclass(slots=True)
class ReliabilityConfig:
    """Configuration for VOS reliability scoring and state classification."""

    w_q: float = 0.30
    w_T: float = 0.20
    w_C: float = 0.25
    w_G: float = 0.15
    w_margin: float = 0.10
    w_A: float = 0.20
    w_E: float = 0.50
    alpha: float = 0.60
    eps: float = 1e-6
    min_area: int = 16
    stable_threshold: float = 0.65
    lost_threshold: float = 0.40
    drift_area_jump_threshold: float = 0.75
    drift_temporal_iou_threshold: float = 0.25
    drift_motion_iou_threshold: float = 0.25
    drift_appearance_threshold: float = 0.35
    drift_candidate_margin_threshold: float = 0.05
    drift_min_reasons: int = 2
    lost_model_quality_threshold: float = 0.20
    rgb_bins: int = 16
    hsv_bins: int = 16
    edge_bins: int = 8


@dataclass(slots=True)
class ReliabilityResult:
    """Full reliability output for one object on one frame."""

    frame_id: str | int
    object_id: int | str
    area: int
    area_jump: float
    temporal_iou: float
    appearance_similarity: float
    motion_consistency: float
    predicted_iou: float
    objectness: float
    candidate_margin: float
    reliability: float
    state: str
    current_bbox: list[float] | None = None
    motion_pred_bbox: list[float] | None = None
    model_quality: float = 0.0
    raw_reliability: float = 0.0
    drift: bool = False
    lost: bool = False
    drift_reasons: list[str] = field(default_factory=list)
    lost_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable full result."""

        return asdict(self)

    def to_debug_dict(self) -> dict[str, Any]:
        """Return the fixed per-frame ``reliability.json`` schema."""

        return {
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "area": self.area,
            "area_jump": self.area_jump,
            "temporal_iou": self.temporal_iou,
            "appearance_similarity": self.appearance_similarity,
            "motion_consistency": self.motion_consistency,
            "predicted_iou": self.predicted_iou,
            "objectness": self.objectness,
            "candidate_margin": self.candidate_margin,
            "reliability": self.reliability,
            "state": self.state,
        }


def _as_config(config: ReliabilityConfig | dict[str, Any] | None) -> ReliabilityConfig:
    """Normalize config input to ``ReliabilityConfig``."""

    if config is None:
        return ReliabilityConfig()
    if isinstance(config, ReliabilityConfig):
        return config
    allowed = ReliabilityConfig.__dataclass_fields__.keys()
    return ReliabilityConfig(**{key: value for key, value in config.items() if key in allowed})


def _as_image_array(image: str | Path | Image.Image | np.ndarray | None) -> np.ndarray | None:
    """Convert an image-like object to RGB uint8 numpy array."""

    if image is None:
        return None
    if isinstance(image, (str, Path)):
        return np.asarray(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] >= 3:
        return array[..., :3].astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {array.shape}")


def _foreground(mask: np.ndarray | Image.Image | str | Path | None, object_id: int | str | None = None) -> np.ndarray:
    """Convert a mask-like object to a boolean foreground array."""

    if mask is None:
        return np.zeros((0, 0), dtype=bool)
    if isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        if object_id is not None and str(object_id).isdigit():
            first_channel = array[..., 0]
            unique_values = set(int(value) for value in np.unique(first_channel).tolist())
            if not unique_values.issubset({0, 1, 255}):
                return first_channel == int(object_id)
        return np.any(array[..., :3] > 0, axis=-1)
    if object_id is not None and str(object_id).isdigit():
        unique_values = set(int(value) for value in np.unique(array).tolist())
        if not unique_values.issubset({0, 1, 255}):
            return array == int(object_id)
    return array > 0


def _bbox_array(box: BBox) -> np.ndarray | None:
    """Normalize a bbox to float array or return None."""

    if box is None:
        return None
    array = np.asarray(box, dtype=np.float32).reshape(-1)
    if array.size != 4 or not np.all(np.isfinite(array)):
        return None
    return array


def _clip01(value: float) -> float:
    """Clip a scalar to [0, 1] after replacing non-finite values."""

    if not math.isfinite(float(value)):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def mask_area(mask: np.ndarray | Image.Image | str | Path, object_id: int | str | None = None) -> int:
    """Return foreground area for a binary or indexed mask."""

    return int(_foreground(mask, object_id).sum())


def mask_to_bbox(mask: np.ndarray | Image.Image | str | Path, object_id: int | str | None = None) -> list[int] | None:
    """Return inclusive ``[xmin, ymin, xmax, ymax]`` bbox for a mask, or None if empty."""

    foreground = _foreground(mask, object_id)
    if foreground.size == 0:
        return None
    ys, xs = np.where(foreground)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_iou(box1: BBox, box2: BBox) -> float:
    """Return IoU for two inclusive bboxes."""

    b1 = _bbox_array(box1)
    b2 = _bbox_array(box2)
    if b1 is None or b2 is None:
        return 0.0
    x0 = max(float(b1[0]), float(b2[0]))
    y0 = max(float(b1[1]), float(b2[1]))
    x1 = min(float(b1[2]), float(b2[2]))
    y1 = min(float(b1[3]), float(b2[3]))
    inter_w = max(0.0, x1 - x0 + 1.0)
    inter_h = max(0.0, y1 - y0 + 1.0)
    inter = inter_w * inter_h
    area1 = max(0.0, float(b1[2] - b1[0] + 1.0)) * max(0.0, float(b1[3] - b1[1] + 1.0))
    area2 = max(0.0, float(b2[2] - b2[0] + 1.0)) * max(0.0, float(b2[3] - b2[1] + 1.0))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def mask_iou(
    m1: np.ndarray | Image.Image | str | Path,
    m2: np.ndarray | Image.Image | str | Path,
    object_id: int | str | None = None,
) -> float:
    """Return IoU between two binary or indexed masks."""

    a = _foreground(m1, object_id)
    b = _foreground(m2, object_id)
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def sigmoid(x: float | None) -> float:
    """Return numerically stable sigmoid; None maps to 0.5."""

    if x is None:
        return 0.5
    value = float(x)
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _crop_slices(mask: np.ndarray, bbox: BBox) -> tuple[slice, slice] | None:
    """Return y/x crop slices for a bbox clipped to mask bounds."""

    box = _bbox_array(bbox)
    if box is None:
        inferred = mask_to_bbox(mask)
        box = _bbox_array(inferred)
    if box is None or mask.size == 0:
        return None
    height, width = mask.shape[:2]
    x0 = int(np.clip(math.floor(float(box[0])), 0, width - 1))
    y0 = int(np.clip(math.floor(float(box[1])), 0, height - 1))
    x1 = int(np.clip(math.ceil(float(box[2])), 0, width - 1))
    y1 = int(np.clip(math.ceil(float(box[3])), 0, height - 1))
    if x1 < x0 or y1 < y0:
        return None
    return slice(y0, y1 + 1), slice(x0, x1 + 1)


def _hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    """Return a normalized histogram for a 1D array."""

    if values.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float32)
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def extract_masked_feature(
    image: str | Path | Image.Image | np.ndarray,
    mask: np.ndarray | Image.Image | str | Path,
    bbox: BBox = None,
    config: ReliabilityConfig | dict[str, Any] | None = None,
    object_id: int | str | None = None,
) -> np.ndarray:
    """Extract a lightweight masked color/texture feature from an image crop."""

    cfg = _as_config(config)
    image_array = _as_image_array(image)
    if image_array is None:
        return np.zeros((cfg.rgb_bins * 3 + cfg.hsv_bins * 3 + cfg.edge_bins + 1,), dtype=np.float32)
    foreground = _foreground(mask, object_id)
    if foreground.shape != image_array.shape[:2]:
        foreground = np.asarray(
            Image.fromarray(foreground.astype(np.uint8)).resize((image_array.shape[1], image_array.shape[0]), Image.Resampling.NEAREST)
        ) > 0
    slices = _crop_slices(foreground, bbox)
    if slices is None:
        return np.zeros((cfg.rgb_bins * 3 + cfg.hsv_bins * 3 + cfg.edge_bins + 1,), dtype=np.float32)
    image_crop = image_array[slices]
    mask_crop = foreground[slices]
    if not np.any(mask_crop):
        return np.zeros((cfg.rgb_bins * 3 + cfg.hsv_bins * 3 + cfg.edge_bins + 1,), dtype=np.float32)

    pixels = image_crop[mask_crop]
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv[mask_crop]
    gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)[mask_crop]
    density = np.asarray([float(mask_crop.mean())], dtype=np.float32)

    parts: list[np.ndarray] = []
    for channel in range(3):
        parts.append(_hist(pixels[:, channel], cfg.rgb_bins, (0.0, 256.0)))
    for channel, value_range in enumerate(((0.0, 180.0), (0.0, 256.0), (0.0, 256.0))):
        parts.append(_hist(hsv_pixels[:, channel], cfg.hsv_bins, value_range))
    parts.append(_hist(edge_mag, cfg.edge_bins, (0.0, 256.0)))
    parts.append(density)
    feature = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else feature


def cosine_similarity(feature_a: np.ndarray | None, feature_b: np.ndarray | None) -> float:
    """Return cosine similarity clipped to [0, 1] for reliability use."""

    if feature_a is None or feature_b is None:
        return 0.0
    a = np.asarray(feature_a, dtype=np.float32).reshape(-1)
    b = np.asarray(feature_b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return _clip01(float(np.dot(a, b) / denom))


def _call_feature_fn(
    feature_fn: FeatureFn,
    image: str | Path | Image.Image | np.ndarray,
    mask: np.ndarray | Image.Image | str | Path,
    bbox: BBox,
    object_id: int | str | None,
    config: ReliabilityConfig,
) -> np.ndarray:
    """Call a user feature function with flexible signatures."""

    try:
        return np.asarray(feature_fn(image=image, mask=mask, bbox=bbox, object_id=object_id, config=config), dtype=np.float32)
    except TypeError:
        try:
            return np.asarray(feature_fn(image, mask, bbox), dtype=np.float32)
        except TypeError:
            return np.asarray(feature_fn(image, mask), dtype=np.float32)


def _appearance_similarity(
    image_t: str | Path | Image.Image | np.ndarray | None,
    mask_t: np.ndarray | Image.Image | str | Path,
    image_0: str | Path | Image.Image | np.ndarray | None,
    mask_0: np.ndarray | Image.Image | str | Path | None,
    current_bbox: BBox,
    object_id: int | str | None,
    config: ReliabilityConfig,
    feature_fn: FeatureFn | None,
    feature_t: np.ndarray | None,
    feature_0: np.ndarray | None,
) -> float:
    """Compute appearance similarity from features, callback, or lightweight default."""

    if feature_t is None and feature_fn is not None and image_t is not None:
        feature_t = _call_feature_fn(feature_fn, image_t, mask_t, current_bbox, object_id, config)
    if feature_0 is None and feature_fn is not None and image_0 is not None and mask_0 is not None:
        feature_0 = _call_feature_fn(feature_fn, image_0, mask_0, None, object_id, config)
    if feature_t is None and image_t is not None:
        feature_t = extract_masked_feature(image_t, mask_t, current_bbox, config, object_id)
    if feature_0 is None and image_0 is not None and mask_0 is not None:
        feature_0 = extract_masked_feature(image_0, mask_0, None, config, object_id)
    return cosine_similarity(feature_t, feature_0)


def classify_state(R_t: float, area: int, config: ReliabilityConfig | dict[str, Any] | None = None) -> str:
    """Classify reliability into stable, ambiguous, or lost."""

    cfg = _as_config(config)
    if area < cfg.min_area or R_t < cfg.lost_threshold:
        return "lost"
    if R_t >= cfg.stable_threshold:
        return "stable"
    return "ambiguous"


def detect_drift(
    result: ReliabilityResult | None = None,
    *,
    area_jump: float | None = None,
    temporal_iou: float | None = None,
    motion_consistency: float | None = None,
    appearance_similarity: float | None = None,
    candidate_margin: float | None = None,
    reliability: float | None = None,
    config: ReliabilityConfig | dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Detect likely drift from reliability components and return reasons."""

    cfg = _as_config(config)
    if result is not None:
        area_jump = result.area_jump
        temporal_iou = result.temporal_iou
        motion_consistency = result.motion_consistency
        appearance_similarity = result.appearance_similarity
        candidate_margin = result.candidate_margin
        reliability = result.reliability
    reasons: list[str] = []
    if area_jump is not None and area_jump > cfg.drift_area_jump_threshold:
        reasons.append("large_area_jump")
    if temporal_iou is not None and temporal_iou < cfg.drift_temporal_iou_threshold:
        reasons.append("low_temporal_iou")
    if motion_consistency is not None and motion_consistency < cfg.drift_motion_iou_threshold:
        reasons.append("low_motion_consistency")
    if appearance_similarity is not None and appearance_similarity < cfg.drift_appearance_threshold:
        reasons.append("low_appearance_similarity")
    if candidate_margin is not None and candidate_margin < cfg.drift_candidate_margin_threshold:
        reasons.append("low_candidate_margin")
    if reliability is not None and cfg.lost_threshold <= reliability < cfg.stable_threshold:
        reasons.append("ambiguous_reliability")
    return len(reasons) >= cfg.drift_min_reasons, reasons


def detect_lost(
    result: ReliabilityResult | None = None,
    *,
    state: str | None = None,
    area: int | None = None,
    reliability: float | None = None,
    model_quality: float | None = None,
    config: ReliabilityConfig | dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Detect lost target state and return reasons."""

    cfg = _as_config(config)
    if result is not None:
        state = result.state
        area = result.area
        reliability = result.reliability
        model_quality = result.model_quality
    reasons: list[str] = []
    if state == "lost":
        reasons.append("classified_lost")
    if area is not None and area < cfg.min_area:
        reasons.append("empty_or_too_small_mask")
    if reliability is not None and reliability < cfg.lost_threshold:
        reasons.append("very_low_reliability")
    if model_quality is not None and model_quality < cfg.lost_model_quality_threshold:
        reasons.append("low_model_quality")
    return bool(reasons), reasons


def compute_reliability(
    image_t: str | Path | Image.Image | np.ndarray,
    mask_t: np.ndarray | Image.Image | str | Path,
    mask_prev: np.ndarray | Image.Image | str | Path,
    image_0: str | Path | Image.Image | np.ndarray | None = None,
    mask_0: np.ndarray | Image.Image | str | Path | None = None,
    bbox_t: BBox = None,
    motion_pred_bbox: BBox = None,
    predicted_iou: float | None = None,
    objectness_logit: float | None = None,
    top1_sim: float | None = None,
    top2_sim: float | None = None,
    config: ReliabilityConfig | dict[str, Any] | None = None,
    frame_id: str | int = "",
    object_id: int | str = 1,
    feature_fn: FeatureFn | None = None,
    feature_t: np.ndarray | None = None,
    feature_0: np.ndarray | None = None,
) -> ReliabilityResult:
    """Compute reliability score and state for one object on one frame."""

    cfg = _as_config(config)
    area = mask_area(mask_t, object_id)
    prev_area = mask_area(mask_prev, object_id)
    area_jump = abs(math.log((area + cfg.eps) / (prev_area + cfg.eps)))
    temporal = _clip01(mask_iou(mask_t, mask_prev, object_id))
    current_bbox = _bbox_array(bbox_t if bbox_t is not None else mask_to_bbox(mask_t, object_id))
    current_bbox_list = [float(value) for value in current_bbox.tolist()] if current_bbox is not None else None
    motion_box = _bbox_array(motion_pred_bbox)
    motion_box_list = [float(value) for value in motion_box.tolist()] if motion_box is not None else None
    motion = _clip01(bbox_iou(current_bbox, motion_box)) if motion_box is not None else 0.0
    appearance = _appearance_similarity(
        image_t=image_t,
        mask_t=mask_t,
        image_0=image_0,
        mask_0=mask_0,
        current_bbox=current_bbox,
        object_id=object_id,
        config=cfg,
        feature_fn=feature_fn,
        feature_t=feature_t,
        feature_0=feature_0,
    )
    pred_iou = _clip01(0.0 if predicted_iou is None else float(predicted_iou))
    objectness = _clip01(sigmoid(objectness_logit))
    model_quality = cfg.alpha * pred_iou + (1.0 - cfg.alpha) * objectness
    margin = float((top1_sim if top1_sim is not None else 0.0) - (top2_sim if top2_sim is not None else 0.0))
    margin_for_score = float(np.clip(margin, -1.0, 1.0))
    empty_penalty = 1.0 if area < cfg.min_area else 0.0
    raw_reliability = (
        cfg.w_q * model_quality
        + cfg.w_T * temporal
        + cfg.w_C * appearance
        + cfg.w_G * motion
        + cfg.w_margin * margin_for_score
        - cfg.w_A * area_jump
        - cfg.w_E * empty_penalty
    )
    reliability = _clip01(raw_reliability)
    state = classify_state(reliability, area, cfg)
    result = ReliabilityResult(
        frame_id=frame_id,
        object_id=object_id,
        area=area,
        area_jump=float(area_jump),
        temporal_iou=float(temporal),
        appearance_similarity=float(appearance),
        motion_consistency=float(motion),
        predicted_iou=float(pred_iou),
        objectness=float(objectness),
        candidate_margin=float(margin),
        reliability=float(reliability),
        state=state,
        current_bbox=current_bbox_list,
        motion_pred_bbox=motion_box_list,
        model_quality=float(model_quality),
        raw_reliability=float(raw_reliability),
    )
    result.drift, result.drift_reasons = detect_drift(result, config=cfg)
    result.lost, result.lost_reasons = detect_lost(result, config=cfg)
    return result


def reliability_to_json(result: ReliabilityResult, output_path: str | Path) -> None:
    """Write fixed-schema per-frame reliability debug JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_debug_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def _draw_box(draw: ImageDraw.ImageDraw, box: list[float] | None, color: tuple[int, int, int], width: int) -> None:
    """Draw a bbox if present."""

    if box is None:
        return
    coords = [int(round(value)) for value in box]
    for offset in range(width):
        draw.rectangle(
            [coords[0] - offset, coords[1] - offset, coords[2] + offset, coords[3] + offset],
            outline=color,
        )


def draw_reliability_overlay(
    image: str | Path | Image.Image | np.ndarray,
    result: ReliabilityResult,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Draw current bbox, motion-predicted bbox, state, and reliability score."""

    image_array = _as_image_array(image)
    if image_array is None:
        raise ValueError("image must not be None")
    overlay = Image.fromarray(image_array.copy()).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    _draw_box(draw, result.current_bbox, (34, 197, 94), 2)
    _draw_box(draw, result.motion_pred_bbox, (59, 130, 246), 2)
    text = f"{result.state} R={result.reliability:.3f}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((8, 8), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3], fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(path)
    return overlay

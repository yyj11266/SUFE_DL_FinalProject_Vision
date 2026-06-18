"""Object-level anchor mining for re-prompting VOS pipelines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.features.dino_features import TargetFeaturePool, extract_augmented_target_pool, extract_crop_feature
from src.vos.reliability import bbox_iou, cosine_similarity, mask_area, mask_to_bbox


@dataclass(slots=True)
class CandidateMask:
    """Candidate object mask generated for one sampled frame."""

    frame_index: int
    frame_id: str
    mask: np.ndarray
    bbox: list[float]
    q: float
    source: str


@dataclass(slots=True)
class AnchorScore:
    """Scored candidate anchor for one object and frame."""

    frame_id: str
    frame_index: int
    bbox: list[float]
    mask_path: str
    S_app: float
    S_mot: float
    S_scale: float
    q: float
    S_anchor: float
    D_dup: float = 0.0
    selected: bool = False
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe anchor output fields."""

        return {
            "frame_id": self.frame_id,
            "bbox": self.bbox,
            "mask_path": self.mask_path,
            "S_app": self.S_app,
            "S_mot": self.S_mot,
            "S_scale": self.S_scale,
            "q": self.q,
            "S_anchor": self.S_anchor,
        }

    def to_debug_dict(self) -> dict[str, Any]:
        """Return extended debug fields."""

        payload = asdict(self)
        return payload


@dataclass(slots=True)
class AnchorMiningConfig:
    """Configuration for object-level anchor mining."""

    sample_stride: int = 8
    top_k_anchors: int = 5
    temporal_nms: int = 8
    anchor_threshold: float = 0.72
    eps: float = 1e-6
    bbox_tracker_q: float = 0.35
    sam_auto_q: float = 0.50
    detector_q_default: float = 0.70
    bbox_padding_ratio: float = 0.08
    debug_fps: float = 6.0


@dataclass(slots=True)
class AnchorMiningResult:
    """Anchor mining result for one video/object pair."""

    video_id: str
    object_id: int | str
    anchors: list[AnchorScore]
    candidates: list[AnchorScore]
    warnings: list[str] = field(default_factory=list)
    debug_video_path: str | None = None
    anchor_json_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe result metadata."""

        return {
            "video_id": self.video_id,
            "object_id": self.object_id,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "candidates": [candidate.to_debug_dict() for candidate in self.candidates],
            "warnings": self.warnings,
            "debug_video_path": self.debug_video_path,
            "anchor_json_path": self.anchor_json_path,
        }


def _as_config(config: AnchorMiningConfig | dict[str, Any] | None) -> AnchorMiningConfig:
    """Normalize config input to ``AnchorMiningConfig``."""

    if config is None:
        return AnchorMiningConfig()
    if isinstance(config, AnchorMiningConfig):
        return config
    allowed = AnchorMiningConfig.__dataclass_fields__.keys()
    return AnchorMiningConfig(**{key: value for key, value in config.items() if key in allowed})


def _natural_key(path: str | Path) -> list[int | str]:
    """Sort strings with embedded numbers in numeric order."""

    import re

    text = path.as_posix() if isinstance(path, Path) else str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _load_image(path_or_image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load image-like input as RGB PIL image."""

    if isinstance(path_or_image, (str, Path)):
        return Image.open(path_or_image).convert("RGB")
    if isinstance(path_or_image, Image.Image):
        return path_or_image.convert("RGB")
    array = np.asarray(path_or_image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return Image.fromarray(array[..., :3].astype(np.uint8)).convert("RGB")


def _load_mask(mask_or_path: str | Path | Image.Image | np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Load mask-like input as boolean foreground mask."""

    if isinstance(mask_or_path, (str, Path)):
        array = np.asarray(Image.open(mask_or_path))
    elif isinstance(mask_or_path, Image.Image):
        array = np.asarray(mask_or_path)
    else:
        array = np.asarray(mask_or_path)
    if array.ndim == 3:
        mask = np.any(array[..., :3] > 0, axis=-1)
    else:
        mask = array > 0
    if size is not None and mask.shape != (size[1], size[0]):
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0
    return mask


def _frame_paths(frames: Iterable[str | Path | Image.Image | np.ndarray]) -> list[str | Path | Image.Image | np.ndarray]:
    """Return frames in stable order where path-like inputs are naturally sorted."""

    items = list(frames)
    if all(isinstance(item, (str, Path)) for item in items):
        return sorted(items, key=_natural_key)
    return items


def _bbox_center(box: list[float]) -> tuple[float, float]:
    """Return bbox center."""

    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _bbox_area(box: list[float] | np.ndarray | None) -> float:
    """Return inclusive bbox area."""

    if box is None:
        return 0.0
    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.size != 4:
        return 0.0
    return max(0.0, float(values[2] - values[0] + 1.0)) * max(0.0, float(values[3] - values[1] + 1.0))


def _clip_box(box: list[float] | np.ndarray, size: tuple[int, int]) -> list[float]:
    """Clip bbox to image size."""

    width, height = size
    values = np.asarray(box, dtype=np.float32).reshape(4)
    values[[0, 2]] = np.clip(values[[0, 2]], 0, width - 1)
    values[[1, 3]] = np.clip(values[[1, 3]], 0, height - 1)
    return [float(value) for value in values.tolist()]


def _pad_box(box: list[float], size: tuple[int, int], ratio: float) -> list[float]:
    """Pad an inclusive bbox by a ratio of width and height."""

    x0, y0, x1, y1 = [float(value) for value in box]
    width = max(1.0, x1 - x0 + 1.0)
    height = max(1.0, y1 - y0 + 1.0)
    return _clip_box([x0 - ratio * width, y0 - ratio * height, x1 + ratio * width, y1 + ratio * height], size)


def _box_mask(box: list[float], size: tuple[int, int]) -> np.ndarray:
    """Create a boolean mask from an inclusive bbox."""

    width, height = size
    x0, y0, x1, y1 = [int(round(value)) for value in _clip_box(box, size)]
    mask = np.zeros((height, width), dtype=bool)
    if x1 >= x0 and y1 >= y0:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _predict_bbox(history: list[tuple[int, list[float]]], frame_index: int, size: tuple[int, int]) -> list[float]:
    """Predict current bbox from one or two previous anchors using linear motion."""

    if not history:
        return [0.0, 0.0, float(size[0] - 1), float(size[1] - 1)]
    if len(history) == 1:
        return _clip_box(history[-1][1], size)
    t1, b1 = history[-1]
    t0, b0 = history[-2]
    step = max(1, t1 - t0)
    factor = (frame_index - t1) / float(step)
    predicted = np.asarray(b1, dtype=np.float32) + factor * (np.asarray(b1, dtype=np.float32) - np.asarray(b0, dtype=np.float32))
    return _clip_box(predicted, size)


def _candidate_from_payload(payload: Any, frame_index: int, frame_id: str, size: tuple[int, int], q: float, source: str) -> list[CandidateMask]:
    """Normalize detector payloads to ``CandidateMask`` entries."""

    if payload is None:
        return []
    if isinstance(payload, dict) and "candidates" in payload:
        payload = payload["candidates"]
    if isinstance(payload, dict):
        payload = [payload]
    candidates: list[CandidateMask] = []
    for item in payload:
        try:
            item_q = float(item.get("q", item.get("score", item.get("confidence", q)))) if isinstance(item, dict) else q
            mask = None
            bbox = None
            if isinstance(item, CandidateMask):
                candidates.append(item)
                continue
            if isinstance(item, dict):
                if "mask" in item:
                    mask = _load_mask(item["mask"], size)
                if "bbox" in item:
                    bbox = _clip_box(item["bbox"], size)
            if mask is None and bbox is not None:
                mask = _box_mask(bbox, size)
            if bbox is None and mask is not None:
                bbox_int = mask_to_bbox(mask)
                bbox = [float(value) for value in bbox_int] if bbox_int is not None else None
            if mask is None or bbox is None or mask_area(mask) <= 0:
                continue
            candidates.append(CandidateMask(frame_index, frame_id, mask, bbox, item_q, source))
        except Exception:
            continue
    return candidates


def _call_detector(detector: Any, image: Image.Image, frame_index: int, visual_prompt: Any, text_prompt: str | None) -> Any:
    """Call optional detector adapters using duck typing."""

    kwargs = {
        "image": image,
        "frame_index": frame_index,
        "visual_prompt": visual_prompt,
        "text_prompt": text_prompt,
    }
    if hasattr(detector, "detect"):
        try:
            return detector.detect(**kwargs)
        except TypeError:
            return detector.detect(image)
    if hasattr(detector, "generate"):
        try:
            return detector.generate(image=image)
        except TypeError:
            return detector.generate(image)
    if callable(detector):
        try:
            return detector(**kwargs)
        except TypeError:
            return detector(image)
    return None


def _generate_detector_candidates(
    detectors: dict[str, Any] | None,
    image: Image.Image,
    frame_index: int,
    frame_id: str,
    size: tuple[int, int],
    visual_prompt: Any,
    text_prompt: str | None,
    config: AnchorMiningConfig,
    warnings: list[str],
) -> list[CandidateMask]:
    """Generate candidates from optional detectors in priority order."""

    detectors = detectors or {}
    priority = [
        ("sam3", config.detector_q_default),
        ("trex2", config.detector_q_default),
        ("groundingdino", config.detector_q_default),
        ("sam_auto", config.sam_auto_q),
        ("sam_automatic_mask_generator", config.sam_auto_q),
    ]
    for name, q in priority:
        detector = detectors.get(name)
        if detector is None:
            warnings.append(f"{name} unavailable; skipping optional detector.")
            continue
        try:
            payload = _call_detector(detector, image, frame_index, visual_prompt, text_prompt)
            candidates = _candidate_from_payload(payload, frame_index, frame_id, size, q, name)
            if candidates:
                return candidates
            warnings.append(f"{name} returned no usable candidates.")
        except Exception as exc:
            warnings.append(f"{name} failed; skipping: {type(exc).__name__}: {exc}")
    return []


def _bbox_tracker_candidates(
    frame_index: int,
    frame_id: str,
    size: tuple[int, int],
    predicted_box: list[float],
    config: AnchorMiningConfig,
) -> list[CandidateMask]:
    """Generate bbox tracker fallback candidates around the predicted box."""

    x0, y0, x1, y1 = [float(value) for value in predicted_box]
    box_w = max(1.0, x1 - x0 + 1.0)
    box_h = max(1.0, y1 - y0 + 1.0)
    offsets = [
        (0.0, 0.0),
        (-0.5 * box_w, 0.0),
        (0.5 * box_w, 0.0),
        (0.0, -0.5 * box_h),
        (0.0, 0.5 * box_h),
        (-0.5 * box_w, -0.5 * box_h),
        (0.5 * box_w, -0.5 * box_h),
        (-0.5 * box_w, 0.5 * box_h),
        (0.5 * box_w, 0.5 * box_h),
    ]
    candidates: list[CandidateMask] = []
    seen: set[tuple[int, int, int, int]] = set()
    for dx, dy in offsets:
        shifted = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
        bbox = _pad_box(shifted, size, config.bbox_padding_ratio)
        key = tuple(int(round(value)) for value in bbox)
        if key in seen:
            continue
        seen.add(key)
        mask = _box_mask(bbox, size)
        candidates.append(CandidateMask(frame_index, frame_id, mask, bbox, config.bbox_tracker_q, "bbox_tracker"))
    return candidates


def _score_candidate(
    candidate: CandidateMask,
    image: Image.Image,
    target_features: np.ndarray,
    predicted_box: list[float],
    selected: list[AnchorScore],
    config: AnchorMiningConfig,
) -> AnchorScore:
    """Score one candidate with appearance, motion, scale, detector quality, and duplicate penalty."""

    feature = extract_crop_feature(image, candidate.mask)
    similarities = [cosine_similarity(feature, target) for target in target_features]
    s_app = max(similarities) if similarities else 0.0
    s_mot = bbox_iou(candidate.bbox, predicted_box)
    scale_ratio = (_bbox_area(candidate.bbox) + config.eps) / (_bbox_area(predicted_box) + config.eps)
    s_scale = math.exp(-abs(math.log(scale_ratio)))
    nearby = [
        anchor
        for anchor in selected
        if abs(anchor.frame_index - candidate.frame_index) < config.temporal_nms
    ]
    d_dup = max((bbox_iou(anchor.bbox, candidate.bbox) for anchor in nearby), default=0.0)
    s_anchor = 0.45 * s_app + 0.20 * s_mot + 0.10 * s_scale + 0.20 * candidate.q - 0.05 * d_dup
    return AnchorScore(
        frame_id=candidate.frame_id,
        frame_index=candidate.frame_index,
        bbox=[float(value) for value in candidate.bbox],
        mask_path="",
        S_app=float(s_app),
        S_mot=float(s_mot),
        S_scale=float(s_scale),
        q=float(candidate.q),
        S_anchor=float(s_anchor),
        D_dup=float(d_dup),
        source=candidate.source,
    )


def _segment_bucket(frame_index: int, frame_count: int) -> int:
    """Map a frame index to one of three temporal buckets."""

    if frame_count <= 1:
        return 0
    return min(2, int((frame_index / max(1, frame_count - 1)) * 3))


def _select_anchors(scores: list[AnchorScore], frame_count: int, config: AnchorMiningConfig) -> list[AnchorScore]:
    """Select anchors using score threshold, temporal NMS, and temporal diversity."""

    eligible = [score for score in scores if score.S_anchor >= config.anchor_threshold]
    eligible.sort(key=lambda item: item.S_anchor, reverse=True)
    selected: list[AnchorScore] = []
    used_buckets: set[int] = set()
    for bucket in range(3):
        bucket_scores = [score for score in eligible if _segment_bucket(score.frame_index, frame_count) == bucket]
        for score in bucket_scores:
            if all(abs(score.frame_index - chosen.frame_index) >= config.temporal_nms for chosen in selected):
                selected.append(score)
                used_buckets.add(bucket)
                break
        if len(selected) >= config.top_k_anchors:
            break
    for score in eligible:
        if len(selected) >= config.top_k_anchors:
            break
        if score in selected:
            continue
        if all(abs(score.frame_index - chosen.frame_index) >= config.temporal_nms for chosen in selected):
            selected.append(score)
    selected.sort(key=lambda item: item.frame_index)
    for score in selected:
        score.selected = True
    return selected


def _save_candidate_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save candidate mask as an 8-bit PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_path)


def _write_anchor_json(path: Path, anchors: list[AnchorScore]) -> None:
    """Write selected anchor JSON list."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([anchor.to_dict() for anchor in anchors], indent=2, ensure_ascii=True), encoding="utf-8")


def _draw_debug_frame(image: Image.Image, scores: list[AnchorScore], selected_paths: set[str]) -> np.ndarray:
    """Draw candidate and selected anchor boxes for one frame."""

    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for score in scores:
        color = (34, 197, 94) if score.mask_path in selected_paths or score.selected else (250, 204, 21)
        box = [int(round(value)) for value in score.bbox]
        draw.rectangle(box, outline=color, width=3 if score.selected else 1)
        draw.text((box[0], max(0, box[1] - 12)), f"{score.S_anchor:.2f}/{score.S_app:.2f}", fill=color, font=font)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def _write_debug_video(
    frames: list[str | Path | Image.Image | np.ndarray],
    scores: list[AnchorScore],
    selected: list[AnchorScore],
    output_path: Path,
    fps: float,
) -> None:
    """Write anchor debug MP4."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_by_frame: dict[int, list[AnchorScore]] = {}
    for score in scores:
        scores_by_frame.setdefault(score.frame_index, []).append(score)
    selected_paths = {score.mask_path for score in selected}
    first = _load_image(frames[0])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        first.size,
    )
    for index, frame in enumerate(frames):
        image = _load_image(frame)
        if image.size != first.size:
            image = image.resize(first.size, Image.Resampling.BILINEAR)
        writer.write(_draw_debug_frame(image, scores_by_frame.get(index, []), selected_paths))
    writer.release()


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    """Return warnings in first-seen order without duplicates."""

    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        if warning in seen:
            continue
        seen.add(warning)
        unique.append(warning)
    return unique


def mine_anchors_for_video(
    frames: Iterable[str | Path | Image.Image | np.ndarray],
    initial_mask: str | Path | Image.Image | np.ndarray,
    output_dir: str | Path,
    target_feature_pool: TargetFeaturePool | np.ndarray | None = None,
    detectors: dict[str, Any] | None = None,
    config: AnchorMiningConfig | dict[str, Any] | None = None,
    video_id: str = "video",
    object_id: int | str = 1,
    text_prompt: str | None = None,
) -> AnchorMiningResult:
    """Mine reliable object anchors for one video/object pair."""

    cfg = _as_config(config)
    frame_items = _frame_paths(frames)
    if not frame_items:
        raise ValueError("frames must contain at least one frame")
    output_root = Path(output_dir)
    anchor_dir = output_root / "anchors" / video_id
    candidate_dir = anchor_dir / "candidate_masks" / str(object_id)
    warnings: list[str] = []
    first_image = _load_image(frame_items[0])
    init_mask = _load_mask(initial_mask, first_image.size)
    initial_bbox = mask_to_bbox(init_mask)
    if initial_bbox is None:
        raise ValueError("initial_mask has no foreground")

    if target_feature_pool is None:
        pool = extract_augmented_target_pool(first_image, init_mask)
        target_features = pool.features
        warnings.extend(pool.backend.get("warnings", []))
    elif isinstance(target_feature_pool, TargetFeaturePool):
        target_features = target_feature_pool.features
        warnings.extend(target_feature_pool.backend.get("warnings", []))
    else:
        target_features = np.asarray(target_feature_pool, dtype=np.float32)
        if target_features.ndim == 1:
            target_features = target_features[None, :]

    history: list[tuple[int, list[float]]] = [(0, [float(value) for value in initial_bbox])]
    scored_candidates: list[AnchorScore] = []
    sampled_indices = list(range(0, len(frame_items), max(1, cfg.sample_stride)))
    if 0 not in sampled_indices:
        sampled_indices.insert(0, 0)

    for frame_index in sampled_indices:
        frame_id = Path(frame_items[frame_index]).stem if isinstance(frame_items[frame_index], (str, Path)) else f"{frame_index:05d}"
        image = _load_image(frame_items[frame_index])
        predicted_box = _predict_bbox(history, frame_index, image.size)
        candidates = _generate_detector_candidates(
            detectors=detectors,
            image=image,
            frame_index=frame_index,
            frame_id=frame_id,
            size=image.size,
            visual_prompt={"mask": init_mask, "bbox": initial_bbox},
            text_prompt=text_prompt,
            config=cfg,
            warnings=warnings,
        )
        if not candidates:
            candidates = _bbox_tracker_candidates(frame_index, frame_id, image.size, predicted_box, cfg)
        frame_scores: list[AnchorScore] = []
        for candidate_index, candidate in enumerate(candidates):
            score = _score_candidate(candidate, image, target_features, predicted_box, scored_candidates, cfg)
            mask_path = candidate_dir / f"{frame_id}_{candidate_index:02d}.png"
            _save_candidate_mask(candidate.mask, mask_path)
            score.mask_path = str(mask_path)
            scored_candidates.append(score)
            frame_scores.append(score)
        if frame_scores:
            best = max(frame_scores, key=lambda item: item.S_anchor)
            history.append((frame_index, best.bbox))
            history = history[-2:]

    anchors = _select_anchors(scored_candidates, len(frame_items), cfg)
    anchor_json_path = anchor_dir / f"{object_id}_anchors.json"
    _write_anchor_json(anchor_json_path, anchors)
    debug_video_path = anchor_dir / "anchor_debug.mp4"
    _write_debug_video(frame_items, scored_candidates, anchors, debug_video_path, cfg.debug_fps)
    return AnchorMiningResult(
        video_id=video_id,
        object_id=object_id,
        anchors=anchors,
        candidates=scored_candidates,
        warnings=_dedupe_warnings(warnings),
        debug_video_path=str(debug_video_path),
        anchor_json_path=str(anchor_json_path),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for anchor mining."""

    parser = argparse.ArgumentParser(description="Mine object-level anchors for SUFE VOS.")
    parser.add_argument("--frames-dir", required=True, help="Directory containing ordered video frames.")
    parser.add_argument("--initial-mask", required=True, help="Initial object mask path.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--video-id", default="video", help="Video id for output paths.")
    parser.add_argument("--object-id", default="1", help="Object id for output paths.")
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--top-k-anchors", type=int, default=5)
    parser.add_argument("--temporal-nms", type=int, default=8)
    parser.add_argument("--anchor-threshold", type=float, default=0.72)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.vos.anchor_mining``."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    frames = sorted(
        [path for path in Path(args.frames_dir).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
        key=_natural_key,
    )
    config = AnchorMiningConfig(
        sample_stride=args.sample_stride,
        top_k_anchors=args.top_k_anchors,
        temporal_nms=args.temporal_nms,
        anchor_threshold=args.anchor_threshold,
    )
    result = mine_anchors_for_video(
        frames=frames,
        initial_mask=args.initial_mask,
        output_dir=args.output_dir,
        config=config,
        video_id=args.video_id,
        object_id=args.object_id,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

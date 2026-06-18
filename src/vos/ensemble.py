"""Optional model ensemble utilities for SUFE VOS predictions."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from src.vos.postprocess import ObjectMaskPrediction, merge_object_predictions, save_indexed_png
from src.vos.reliability import mask_iou


SUPPORTED_PREDICTION_SET_NAMES = {
    "sam2_baseline",
    "sam2_memory_tree",
    "sam3_optional",
    "cutie_optional",
    "xmem_optional",
    "deaot_optional",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(slots=True)
class PredictionSetConfig:
    """Configuration for one model prediction set."""

    name: str
    root: str | Path
    mask_root: str | Path | None = None
    logit_root: str | Path | None = None
    reliability_root: str | Path | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe configuration dictionary."""

        payload = asdict(self)
        for key in ("root", "mask_root", "logit_root", "reliability_root"):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        return payload


@dataclass(slots=True)
class ModelFramePrediction:
    """One model's object-level prediction for a frame."""

    model_name: str
    video_id: str
    frame_stem: str
    frame_index: int
    object_id: int
    mask: np.ndarray | None = None
    logit: np.ndarray | None = None
    reliability: float = 0.5
    drift: bool = False
    state: str = "unknown"
    mask_path: str | None = None
    logit_path: str | None = None
    reliability_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """Return whether this prediction has a mask or logit."""

        return self.mask is not None or self.logit is not None


@dataclass(slots=True)
class EnsembleConfig:
    """Configuration for reliability-weighted optional model ensemble."""

    beta: float = 4.0
    sam3_stable_threshold: float = 0.70
    cutie_reliability_threshold: float = 0.60
    union_iou_threshold: float = 0.65
    uncertain_reliability_threshold: float = 0.45
    intersection_large_recent_ratio: float = 2.0
    intersection_large_image_ratio: float = 0.15
    logit_threshold: float = 0.0
    missing_reliability: float = 0.5
    missing_prediction_policy: str = "empty"


@dataclass(slots=True)
class EnsembleResult:
    """Result for one fused frame."""

    indexed_mask: np.ndarray
    object_masks: dict[int, np.ndarray]
    object_logits: dict[int, np.ndarray]
    strategies: dict[int, str]
    debug_rows: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoEnsembleResult:
    """Result for one ensembled video."""

    video_id: str
    status: str
    frame_count: int
    mask_paths: list[str] = field(default_factory=list)
    object_mask_paths: list[str] = field(default_factory=list)
    debug_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result dictionary."""

        return asdict(self)


def parse_prediction_set_config(value: str) -> PredictionSetConfig:
    """Parse ``name=/path`` or ``/path`` CLI syntax into a prediction set config."""

    if "=" in value:
        name, root = value.split("=", 1)
        name = name.strip()
    else:
        root = value
        name = Path(root).expanduser().name
    if not name:
        raise ValueError(f"Invalid prediction root spec: {value!r}")
    return PredictionSetConfig(name=name, root=Path(root).expanduser())


def normalize_prediction_sets(configs: Iterable[PredictionSetConfig | dict[str, Any] | str]) -> list[PredictionSetConfig]:
    """Normalize prediction set inputs and disable missing roots without failing."""

    normalized: list[PredictionSetConfig] = []
    for item in configs:
        if isinstance(item, PredictionSetConfig):
            cfg = item
        elif isinstance(item, str):
            cfg = parse_prediction_set_config(item)
        else:
            cfg = PredictionSetConfig(**item)
        root = Path(cfg.root).expanduser()
        enabled = bool(cfg.enabled and root.exists())
        normalized.append(
            PredictionSetConfig(
                name=str(cfg.name),
                root=root,
                mask_root=cfg.mask_root,
                logit_root=cfg.logit_root,
                reliability_root=cfg.reliability_root,
                enabled=enabled,
            )
        )
    return normalized


def collect_object_ids_for_video(
    video_id: str,
    prediction_sets: Sequence[PredictionSetConfig],
    data_info: Mapping[str, Any] | None = None,
) -> list[int]:
    """Collect object ids from prompts first, then prediction masks, then fallback to object ``1``."""

    ids: set[int] = set()
    if data_info is not None:
        for video in data_info.get("videos", []):
            if str(video.get("video_id")) != str(video_id):
                continue
            for prompt in video.get("prompts", []):
                for object_id in prompt.get("object_ids", []) or []:
                    if int(object_id) > 0:
                        ids.add(int(object_id))
    if ids:
        return sorted(ids)

    for cfg in prediction_sets:
        if not cfg.enabled:
            continue
        mask_root = _resolve_mask_root(cfg)
        video_dir = mask_root / str(video_id)
        if not video_dir.exists():
            continue
        for mask_path in sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            array = _load_indexed_mask(mask_path)
            positives = [int(value) for value in np.unique(array).tolist() if int(value) > 0]
            if set(positives).issubset({1, 255}) and positives:
                ids.add(1)
            else:
                ids.update(value for value in positives if 0 < value <= 255)
    return sorted(ids) if ids else [1]


def load_model_frame_prediction(
    prediction_set: PredictionSetConfig,
    video_id: str,
    frame_stem: str,
    frame_index: int,
    object_id: int,
    expected_size: tuple[int, int],
    config: EnsembleConfig | dict[str, Any] | None = None,
) -> ModelFramePrediction:
    """Load one model's object-level prediction for a frame/object pair."""

    cfg = _as_config(config)
    mask_path = _find_first_existing(_mask_candidates(prediction_set, video_id, frame_stem))
    logit_path = _find_first_existing(_logit_candidates(prediction_set, video_id, frame_stem))
    reliability, drift, state, reliability_path = _load_reliability(prediction_set, video_id, frame_stem, object_id, cfg)

    mask: np.ndarray | None = None
    logit: np.ndarray | None = None
    if mask_path is not None:
        indexed = _load_indexed_mask(mask_path)
        mask = _object_mask_from_indexed(indexed, object_id)
        mask = _resize_mask(mask, expected_size)
    if logit_path is not None:
        logit = _load_logit_for_object(logit_path, object_id)
        if logit is not None:
            logit = _resize_float(logit, expected_size)
            if mask is None:
                mask = logit > cfg.logit_threshold

    return ModelFramePrediction(
        model_name=prediction_set.name,
        video_id=str(video_id),
        frame_stem=str(frame_stem),
        frame_index=int(frame_index),
        object_id=int(object_id),
        mask=mask,
        logit=logit,
        reliability=float(reliability),
        drift=bool(drift),
        state=str(state),
        mask_path=str(mask_path) if mask_path is not None else None,
        logit_path=str(logit_path) if logit_path is not None else None,
        reliability_path=str(reliability_path) if reliability_path is not None else None,
    )


def ensemble_frame(
    video_id: str,
    frame_stem: str,
    frame_index: int,
    object_ids: Sequence[int],
    prediction_sets: Sequence[PredictionSetConfig],
    expected_size: tuple[int, int],
    recent_areas_by_object: Mapping[int, Sequence[float]] | None = None,
    previous_indexed_mask: np.ndarray | None = None,
    format_spec: Any = None,
    config: EnsembleConfig | dict[str, Any] | None = None,
) -> EnsembleResult:
    """Fuse all available model predictions for one frame into an indexed mask."""

    cfg = _as_config(config)
    object_masks: dict[int, np.ndarray] = {}
    object_logits: dict[int, np.ndarray] = {}
    object_predictions: list[ObjectMaskPrediction] = []
    strategies: dict[int, str] = {}
    debug_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for object_id in object_ids:
        model_predictions = [
            load_model_frame_prediction(pred_set, video_id, frame_stem, frame_index, int(object_id), expected_size, cfg)
            for pred_set in prediction_sets
            if pred_set.enabled
        ]
        fused_mask, fused_logit, strategy, reliability, row, object_warnings = _fuse_object_predictions(
            model_predictions,
            expected_size,
            recent_areas=(recent_areas_by_object or {}).get(int(object_id), []),
            config=cfg,
        )
        warnings.extend(object_warnings)
        object_masks[int(object_id)] = fused_mask
        object_logits[int(object_id)] = fused_logit
        strategies[int(object_id)] = strategy
        debug_row = {
            "video_id": str(video_id),
            "frame_stem": str(frame_stem),
            "frame_index": int(frame_index),
            "object_id": int(object_id),
            **row,
        }
        debug_rows.append(debug_row)
        object_predictions.append(
            ObjectMaskPrediction(
                object_id=int(object_id),
                mask=fused_mask,
                logit=fused_logit,
                reliability=float(reliability),
                bbox=None,
                previous_bbox=None,
                area_initial=float(max(1, fused_mask.sum())),
                target_type="regular",
            )
        )

    merge = merge_object_predictions(object_predictions, previous_indexed_mask=previous_indexed_mask, format_spec=format_spec)
    warnings.extend(merge.warnings)
    return EnsembleResult(
        indexed_mask=merge.indexed_mask,
        object_masks=object_masks,
        object_logits=object_logits,
        strategies=strategies,
        debug_rows=debug_rows,
        warnings=warnings,
    )


def run_ensemble_for_video(
    video_id: str,
    expected_masks: Sequence[Mapping[str, Any]],
    object_ids: Sequence[int],
    prediction_sets: Sequence[PredictionSetConfig],
    output_dir: str | Path,
    format_spec: Any = None,
    data_info: Mapping[str, Any] | None = None,
    skip_existing: bool = False,
    config: EnsembleConfig | dict[str, Any] | None = None,
) -> VideoEnsembleResult:
    """Run ensemble fusion for one video and write masks/debug-ready object masks."""

    del data_info
    cfg = _as_config(config)
    out_dir = Path(output_dir)
    masks_dir = out_dir / "masks" / str(video_id)
    object_dir = out_dir / "object_masks" / str(video_id)
    masks_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)
    previous_indexed: np.ndarray | None = None
    recent_areas: dict[int, list[float]] = {int(object_id): [] for object_id in object_ids}
    mask_paths: list[str] = []
    object_mask_paths: list[str] = []
    debug_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    try:
        ordered_masks = sorted(expected_masks, key=lambda item: (int(item.get("frame_index", 0)), str(item.get("frame_stem", ""))))
        for expected in ordered_masks:
            frame_stem = str(expected.get("frame_stem") or Path(str(expected.get("relative_path", ""))).stem)
            frame_index = int(expected.get("frame_index", len(mask_paths)))
            expected_size = _expected_size(expected, format_spec, video_id)
            output_path = masks_dir / f"{frame_stem}.png"
            if skip_existing and output_path.exists() and _image_has_size(output_path, expected_size):
                previous_indexed = _load_indexed_mask(output_path)
                mask_paths.append(str(output_path))
                for object_id in object_ids:
                    recent_areas.setdefault(int(object_id), []).append(float((previous_indexed == int(object_id)).sum()))
                debug_rows.append(
                    {
                        "video_id": str(video_id),
                        "frame_stem": frame_stem,
                        "frame_index": frame_index,
                        "object_id": "",
                        "strategy": "skipped_existing",
                    }
                )
                continue

            result = ensemble_frame(
                video_id=str(video_id),
                frame_stem=frame_stem,
                frame_index=frame_index,
                object_ids=object_ids,
                prediction_sets=prediction_sets,
                expected_size=expected_size,
                recent_areas_by_object=recent_areas,
                previous_indexed_mask=previous_indexed,
                format_spec=format_spec,
                config=cfg,
            )
            save_indexed_png(result.indexed_mask, output_path, format_spec=format_spec)
            mask_paths.append(str(output_path))
            previous_indexed = result.indexed_mask
            warnings.extend(result.warnings)
            debug_rows.extend(result.debug_rows)
            for object_id, object_mask in result.object_masks.items():
                recent_areas.setdefault(int(object_id), []).append(float(object_mask.sum()))
                obj_path = object_dir / str(object_id) / f"{frame_stem}.png"
                obj_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((object_mask.astype(np.uint8) * 255), mode="L").save(obj_path)
                object_mask_paths.append(str(obj_path))

        return VideoEnsembleResult(
            video_id=str(video_id),
            status="done",
            frame_count=len(ordered_masks),
            mask_paths=mask_paths,
            object_mask_paths=object_mask_paths,
            debug_rows=debug_rows,
            warnings=warnings,
        )
    except Exception as exc:
        return VideoEnsembleResult(
            video_id=str(video_id),
            status="failed",
            frame_count=len(expected_masks),
            mask_paths=mask_paths,
            object_mask_paths=object_mask_paths,
            debug_rows=debug_rows,
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}",
        )


def write_ensemble_debug_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    """Write per-frame/per-object ensemble decisions to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id",
        "frame_stem",
        "frame_index",
        "object_id",
        "strategy",
        "selected_model",
        "model_names",
        "reliabilities",
        "weights",
        "has_logits",
        "output_area",
        "pairwise_min_iou",
        "avg_reliability",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _fuse_object_predictions(
    predictions: Sequence[ModelFramePrediction],
    expected_size: tuple[int, int],
    recent_areas: Sequence[float],
    config: EnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, str, float, dict[str, Any], list[str]]:
    """Fuse predictions for one object and return mask/logit/debug tuple."""

    available = [prediction for prediction in predictions if prediction.available]
    warnings: list[str] = []
    if not available:
        if config.missing_prediction_policy != "empty":
            warnings.append(f"Unknown missing_prediction_policy={config.missing_prediction_policy}; used empty.")
        empty_logit = np.full(expected_size, -1.0, dtype=np.float32)
        empty_mask = np.zeros(expected_size, dtype=bool)
        return empty_mask, empty_logit, "no_prediction_empty", 0.0, _debug_row("no_prediction_empty", [], {}, empty_mask), warnings

    sam3 = _best_named_prediction(available, "sam3")
    if sam3 is not None and sam3.reliability >= config.sam3_stable_threshold:
        mask, logit = _prediction_mask_logit(sam3, expected_size, config)
        row = _debug_row("prefer_sam3_when_stable", available, {sam3.model_name: 1.0}, mask, selected_model=sam3.model_name)
        return mask, logit, "prefer_sam3_when_stable", sam3.reliability, row, warnings

    cutie = _best_named_prediction(available, "cutie")
    if cutie is not None and _sam_family_drift_detected(available) and cutie.reliability >= config.cutie_reliability_threshold:
        mask, logit = _prediction_mask_logit(cutie, expected_size, config)
        row = _debug_row("prefer_cutie_on_drift", available, {cutie.model_name: 1.0}, mask, selected_model=cutie.model_name)
        return mask, logit, "prefer_cutie_on_drift", cutie.reliability, row, warnings

    pairwise_min_iou = _pairwise_min_iou(available, expected_size, config)
    if len(available) >= 2 and pairwise_min_iou >= config.union_iou_threshold:
        mask = np.zeros(expected_size, dtype=bool)
        for prediction in available:
            mask |= _prediction_binary_mask(prediction, expected_size, config)
        logit = mask.astype(np.float32) - 0.5
        reliability = float(np.mean([prediction.reliability for prediction in available]))
        row = _debug_row("union_when_high_agreement", available, _uniform_weights(available), mask)
        row["pairwise_min_iou"] = pairwise_min_iou
        return mask, logit, "union_when_high_agreement", reliability, row, warnings

    avg_reliability = float(np.mean([prediction.reliability for prediction in available]))
    if avg_reliability < config.uncertain_reliability_threshold and _large_uncertain_masks(available, expected_size, recent_areas, config):
        masks = [_prediction_binary_mask(prediction, expected_size, config) for prediction in available]
        intersection = np.logical_and.reduce(masks) if masks else np.zeros(expected_size, dtype=bool)
        if intersection.any():
            logit = intersection.astype(np.float32) - 0.5
            row = _debug_row("intersection_when_uncertain", available, _uniform_weights(available), intersection)
            row["pairwise_min_iou"] = pairwise_min_iou
            return intersection, logit, "intersection_when_uncertain", avg_reliability, row, warnings

    mask, logit, weights = _weighted_vote(available, expected_size, config)
    reliability = float(sum(prediction.reliability * weights[prediction.model_name] for prediction in available))
    row = _debug_row("majority_vote", available, weights, mask)
    row["pairwise_min_iou"] = pairwise_min_iou
    return mask, logit, "majority_vote", reliability, row, warnings


def _weighted_vote(
    predictions: Sequence[ModelFramePrediction],
    expected_size: tuple[int, int],
    config: EnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fuse predictions by reliability softmax and logits/mask probabilities."""

    weights = _softmax_weights(predictions, config.beta)
    use_logits = any(prediction.logit is not None for prediction in predictions)
    if use_logits:
        fused = np.zeros(expected_size, dtype=np.float32)
        for prediction in predictions:
            _, logit = _prediction_mask_logit(prediction, expected_size, config)
            fused += float(weights[prediction.model_name]) * logit
        return fused > config.logit_threshold, fused.astype(np.float32), weights
    prob = np.zeros(expected_size, dtype=np.float32)
    for prediction in predictions:
        prob += float(weights[prediction.model_name]) * _prediction_binary_mask(prediction, expected_size, config).astype(np.float32)
    logit = prob - 0.5
    return logit > config.logit_threshold, logit.astype(np.float32), weights


def _prediction_mask_logit(
    prediction: ModelFramePrediction,
    expected_size: tuple[int, int],
    config: EnsembleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a prediction's mask and signed logit in expected size."""

    if prediction.logit is not None:
        logit = _resize_float(prediction.logit, expected_size)
        return logit > config.logit_threshold, logit.astype(np.float32)
    mask = _prediction_binary_mask(prediction, expected_size, config)
    return mask, mask.astype(np.float32) - 0.5


def _prediction_binary_mask(
    prediction: ModelFramePrediction,
    expected_size: tuple[int, int],
    config: EnsembleConfig,
) -> np.ndarray:
    """Return a boolean mask for a model prediction."""

    if prediction.mask is not None:
        return _resize_mask(prediction.mask, expected_size)
    if prediction.logit is not None:
        return _resize_float(prediction.logit, expected_size) > config.logit_threshold
    return np.zeros(expected_size, dtype=bool)


def _debug_row(
    strategy: str,
    predictions: Sequence[ModelFramePrediction],
    weights: Mapping[str, float],
    output_mask: np.ndarray,
    selected_model: str = "",
) -> dict[str, Any]:
    """Build one CSV-ready debug row."""

    names = [prediction.model_name for prediction in predictions]
    return {
        "strategy": strategy,
        "selected_model": selected_model,
        "model_names": names,
        "reliabilities": {prediction.model_name: round(float(prediction.reliability), 6) for prediction in predictions},
        "weights": {name: round(float(value), 6) for name, value in weights.items()},
        "has_logits": {prediction.model_name: prediction.logit is not None for prediction in predictions},
        "output_area": int(np.asarray(output_mask, dtype=bool).sum()),
        "avg_reliability": float(np.mean([prediction.reliability for prediction in predictions])) if predictions else 0.0,
        "pairwise_min_iou": "",
        "warnings": "",
    }


def _best_named_prediction(predictions: Sequence[ModelFramePrediction], needle: str) -> ModelFramePrediction | None:
    """Return the highest-reliability prediction whose model name contains ``needle``."""

    candidates = [prediction for prediction in predictions if needle.lower() in prediction.model_name.lower()]
    if not candidates:
        return None
    return max(candidates, key=lambda prediction: float(prediction.reliability))


def _sam_family_drift_detected(predictions: Sequence[ModelFramePrediction]) -> bool:
    """Return whether SAM2/SAM3 predictions indicate drift or unstable state."""

    for prediction in predictions:
        name = prediction.model_name.lower()
        if "sam2" not in name and "sam3" not in name:
            continue
        if prediction.drift or prediction.state in {"ambiguous", "lost", "recovery"}:
            return True
    return False


def _pairwise_min_iou(
    predictions: Sequence[ModelFramePrediction],
    expected_size: tuple[int, int],
    config: EnsembleConfig,
) -> float:
    """Return the minimum non-empty pairwise IoU among prediction masks."""

    if len(predictions) < 2:
        return 1.0
    masks = [_prediction_binary_mask(prediction, expected_size, config) for prediction in predictions]
    values: list[float] = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if not masks[i].any() and not masks[j].any():
                continue
            values.append(float(mask_iou(masks[i], masks[j])))
    return min(values) if values else 0.0


def _large_uncertain_masks(
    predictions: Sequence[ModelFramePrediction],
    expected_size: tuple[int, int],
    recent_areas: Sequence[float],
    config: EnsembleConfig,
) -> bool:
    """Return whether low-reliability masks are large enough to prefer intersection."""

    union = np.zeros(expected_size, dtype=bool)
    for prediction in predictions:
        union |= _prediction_binary_mask(prediction, expected_size, config)
    area = float(union.sum())
    if recent_areas:
        median = float(np.median(np.asarray(recent_areas, dtype=np.float32)))
        return bool(median > 0 and area > config.intersection_large_recent_ratio * median)
    image_area = float(max(1, expected_size[0] * expected_size[1]))
    return bool(area / image_area > config.intersection_large_image_ratio)


def _softmax_weights(predictions: Sequence[ModelFramePrediction], beta: float) -> dict[str, float]:
    """Compute reliability softmax weights ``exp(beta*R)`` over model predictions."""

    if not predictions:
        return {}
    scores = np.asarray([float(prediction.reliability) * float(beta) for prediction in predictions], dtype=np.float64)
    scores -= float(scores.max())
    probs = np.exp(scores)
    probs /= max(float(probs.sum()), 1e-12)
    return {prediction.model_name: float(probs[index]) for index, prediction in enumerate(predictions)}


def _uniform_weights(predictions: Sequence[ModelFramePrediction]) -> dict[str, float]:
    """Return uniform model weights for direct union/intersection strategies."""

    if not predictions:
        return {}
    value = 1.0 / float(len(predictions))
    return {prediction.model_name: value for prediction in predictions}


def _resolve_mask_root(config: PredictionSetConfig) -> Path:
    """Resolve mask root from an experiment root or direct masks directory."""

    if config.mask_root is not None:
        return Path(config.mask_root).expanduser()
    root = Path(config.root).expanduser()
    candidate = root / "masks"
    if candidate.exists():
        return candidate
    return root


def _resolve_logit_root(config: PredictionSetConfig) -> Path:
    """Resolve raw-logit root from an experiment root or direct logits directory."""

    if config.logit_root is not None:
        return Path(config.logit_root).expanduser()
    root = Path(config.root).expanduser()
    candidate = root / "raw_logits"
    if candidate.exists():
        return candidate
    return root / "raw_logits"


def _resolve_reliability_root(config: PredictionSetConfig) -> Path:
    """Resolve reliability root from an experiment root or direct reliability directory."""

    if config.reliability_root is not None:
        return Path(config.reliability_root).expanduser()
    root = Path(config.root).expanduser()
    candidate = root / "reliability"
    if candidate.exists():
        return candidate
    return root / "reliability"


def _mask_candidates(config: PredictionSetConfig, video_id: str, frame_stem: str) -> list[Path]:
    """Return plausible indexed mask paths for a frame."""

    mask_root = _resolve_mask_root(config)
    candidates: list[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        candidates.append(mask_root / str(video_id) / f"{frame_stem}{suffix}")
    candidates.append(mask_root / "Annotations" / str(video_id) / f"{frame_stem}.png")
    return candidates


def _logit_candidates(config: PredictionSetConfig, video_id: str, frame_stem: str) -> list[Path]:
    """Return plausible raw logit paths for a frame."""

    logit_root = _resolve_logit_root(config)
    return [
        logit_root / str(video_id) / f"{frame_stem}.npz",
        logit_root / str(video_id) / f"{frame_stem}.npy",
    ]


def _reliability_candidates(config: PredictionSetConfig, video_id: str, frame_stem: str, object_id: int) -> list[Path]:
    """Return plausible reliability JSON paths for a frame/object pair."""

    root = _resolve_reliability_root(config)
    return [
        root / str(video_id) / str(object_id) / f"{frame_stem}.json",
        root / str(video_id) / f"{object_id}" / f"{frame_stem}.json",
        root / str(video_id) / f"{frame_stem}_{object_id}.json",
        root / str(video_id) / f"{frame_stem}.json",
    ]


def _find_first_existing(paths: Iterable[Path]) -> Path | None:
    """Return the first existing path from candidates."""

    for path in paths:
        if path.exists():
            return path
    return None


def _load_reliability(
    config: PredictionSetConfig,
    video_id: str,
    frame_stem: str,
    object_id: int,
    ensemble_config: EnsembleConfig,
) -> tuple[float, bool, str, Path | None]:
    """Load reliability, drift, and state from JSON if present."""

    path = _find_first_existing(_reliability_candidates(config, video_id, frame_stem, object_id))
    if path is None:
        return float(ensemble_config.missing_reliability), False, "unknown", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return float(ensemble_config.missing_reliability), False, "unknown", path
    if isinstance(payload, dict) and "objects" in payload:
        obj_payload = payload.get("objects", {}).get(str(object_id)) or payload.get("objects", {}).get(int(object_id))
        if isinstance(obj_payload, dict):
            payload = obj_payload
    reliability = _optional_float(payload.get("reliability", payload.get("R_t", ensemble_config.missing_reliability)), ensemble_config.missing_reliability)
    state = str(payload.get("state", "unknown"))
    drift = bool(
        payload.get("drift", False)
        or payload.get("drift_detected", False)
        or payload.get("is_drift", False)
        or bool(payload.get("drift_reasons", []))
    )
    return float(reliability), drift, state, path


def _load_indexed_mask(path: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an indexed mask without binarizing object ids."""

    if isinstance(path, np.ndarray):
        array = path
    elif isinstance(path, Image.Image):
        array = np.asarray(path)
    else:
        array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint8)


def _object_mask_from_indexed(indexed: np.ndarray, object_id: int) -> np.ndarray:
    """Extract an object mask from indexed or binary-style masks."""

    array = np.asarray(indexed)
    positives = [int(value) for value in np.unique(array).tolist() if int(value) > 0]
    if not positives:
        return np.zeros(array.shape[:2], dtype=bool)
    if set(positives).issubset({1, 255}):
        return array > 0 if int(object_id) == 1 else np.zeros(array.shape[:2], dtype=bool)
    return array == int(object_id)


def _load_logit_for_object(path: str | Path, object_id: int) -> np.ndarray | None:
    """Load an object logit plane from npz/npy logits."""

    logit_path = Path(path)
    if logit_path.suffix.lower() == ".npy":
        array = np.load(logit_path)
        return _select_logit_plane(array, object_id, None)
    with np.load(logit_path) as payload:
        key = "logits" if "logits" in payload.files else ("logit" if "logit" in payload.files else sorted(payload.files)[0])
        logits = payload[key]
        object_ids = payload["object_ids"] if "object_ids" in payload.files else None
        return _select_logit_plane(logits, object_id, object_ids)


def _select_logit_plane(logits: np.ndarray, object_id: int, object_ids: np.ndarray | None) -> np.ndarray | None:
    """Select one object plane from a logits tensor."""

    array = np.asarray(logits, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3:
        index = 0
        if object_ids is not None:
            ids = [int(value) for value in np.asarray(object_ids).reshape(-1).tolist()]
            if int(object_id) not in ids:
                return None
            index = ids.index(int(object_id))
        elif 0 <= int(object_id) - 1 < array.shape[0]:
            index = int(object_id) - 1
        if 0 <= index < array.shape[0]:
            return array[index].astype(np.float32)
    return None


def _expected_size(expected_mask: Mapping[str, Any], format_spec: Any, video_id: str) -> tuple[int, int]:
    """Return expected size as ``(height, width)``."""

    width = int(expected_mask.get("width") or 0)
    height = int(expected_mask.get("height") or 0)
    if (not width or not height) and format_spec is not None:
        sizes = getattr(format_spec, "image_size_per_video", None)
        if sizes is None and isinstance(format_spec, Mapping):
            sizes = format_spec.get("image_size_per_video")
        if sizes and str(video_id) in sizes:
            width, height = int(sizes[str(video_id)][0]), int(sizes[str(video_id)][1])
    if not width or not height:
        raise ValueError(f"Missing expected width/height for {video_id}/{expected_mask.get('frame_stem')}")
    return int(height), int(width)


def _image_has_size(path: Path, size_hw: tuple[int, int]) -> bool:
    """Return whether an image exists and has the expected size."""

    try:
        with Image.open(path) as image:
            return image.size == (int(size_hw[1]), int(size_hw[0]))
    except Exception:
        return False


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a mask to ``shape`` using nearest interpolation."""

    array = np.asarray(mask)
    if array.shape[:2] == shape:
        return array.astype(bool)
    resized = Image.fromarray(array.astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a float array to ``shape`` using bilinear interpolation."""

    data = np.asarray(array, dtype=np.float32)
    if data.shape[:2] == shape:
        return data.astype(np.float32)
    return cv2.resize(data, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _as_config(config: EnsembleConfig | dict[str, Any] | None) -> EnsembleConfig:
    """Normalize ensemble config input."""

    if config is None:
        return EnsembleConfig()
    if isinstance(config, EnsembleConfig):
        return config
    allowed = EnsembleConfig.__dataclass_fields__.keys()
    return EnsembleConfig(**{key: value for key, value in config.items() if key in allowed})


def _optional_float(value: Any, default: float) -> float:
    """Parse a finite float with fallback."""

    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if math.isfinite(parsed):
        return parsed
    return float(default)


def _csv_value(value: Any) -> str | int | float:
    """Return a CSV-safe scalar value."""

    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)

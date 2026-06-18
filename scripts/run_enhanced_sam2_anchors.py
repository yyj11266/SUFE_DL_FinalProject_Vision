"""Run conservative SAM2 pseudo-anchor repropagation and fusion.

This script is intentionally separate from ``run_baseline_sam2.py``.  It takes
an existing SAM2 baseline experiment, chooses high-stability pseudo anchors from
the baseline masks, reruns SAM2 from those anchor frames, and fuses the original
and anchor predictions into a new submission-ready mask tree.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspect_sufe import inspect_dataset, save_data_info
from src.data.submission import (
    FormatSpec,
    infer_provisional_format,
    inspect_sample_submission,
    make_submission,
    save_format_spec,
    validate_submission_zip,
)
from src.trackers.sam2_tracker import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_MODEL_CFG,
    build_sam2_video_predictor,
    download_sam2_checkpoint,
    install_or_check_sam2,
    run_sam2_on_video,
)
from src.vos.postprocess import ObjectMaskPrediction, PostprocessConfig, postprocess_object_mask
from src.vos.reliability import bbox_iou, mask_to_bbox


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(slots=True)
class AnchorCandidate:
    """One selected pseudo-anchor frame for a video."""

    video_id: str
    frame_index: int
    frame_stem: str
    quality: float
    source_mask_path: str
    run_dir: str | None = None
    status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""

        return {
            "video_id": self.video_id,
            "frame_index": int(self.frame_index),
            "frame_stem": self.frame_stem,
            "quality": float(self.quality),
            "source_mask_path": self.source_mask_path,
            "run_dir": self.run_dir,
            "status": self.status,
            "warnings": self.warnings,
            "error": self.error,
        }


@dataclass(slots=True)
class VideoEnhanceResult:
    """Result metadata for one enhanced video."""

    video_id: str
    status: str
    frame_count: int
    object_ids: list[int]
    anchors: list[AnchorCandidate] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata."""

        return {
            "video_id": self.video_id,
            "status": self.status,
            "frame_count": int(self.frame_count),
            "object_ids": [int(value) for value in self.object_ids],
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "mask_paths": self.mask_paths,
            "warnings": self.warnings,
            "error": self.error,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""

    parser = argparse.ArgumentParser(description="Enhance a SAM2 baseline with pseudo-anchor repropagation.")
    parser.add_argument("--data-root", required=True, help="Extracted SUFE data root.")
    parser.add_argument("--baseline-exp", required=True, help="Existing baseline experiment directory containing masks/.")
    parser.add_argument("--output-dir", required=True, help="Outputs root directory.")
    parser.add_argument("--experiment-id", default=f"sam2_anchor_fusion_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--sample-submission", default=None, help="Optional sample_submission.zip path.")
    parser.add_argument("--data-info", default=None, help="Optional data_info.json. Defaults to baseline-exp/data_info.json.")
    parser.add_argument("--format-spec", default=None, help="Optional format_spec.json. Defaults to baseline-exp/format_spec.json.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_NAME, help="SAM2 checkpoint path or known name.")
    parser.add_argument("--sam2-repo-dir", default=None, help="Reusable official SAM2 repo directory; prefer Colab local disk.")
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG, help="SAM2 model cfg.")
    parser.add_argument("--prompt-mode", default="mask", choices=["mask", "box", "points", "box_points", "mask_box_points"])
    parser.add_argument("--resize-long-side", type=int, default=0, help="SAM2 inference resize long side; 0 keeps original.")
    parser.add_argument("--anchor-fractions", default="0.50", help="Comma-separated target anchor fractions, e.g. 0.50,0.75.")
    parser.add_argument("--anchor-search-radius", type=int, default=8, help="Search radius around each target fraction.")
    parser.add_argument("--max-anchor-runs", type=int, default=1, help="Maximum pseudo-anchor SAM2 reruns per video.")
    parser.add_argument("--anchor-quality-threshold", type=float, default=0.25, help="Skip anchors below this self-stability score.")
    parser.add_argument("--fusion-tau", type=float, default=48.0, help="Temporal decay constant for anchor fusion.")
    parser.add_argument("--baseline-floor", type=float, default=0.35, help="Minimum baseline vote weight.")
    parser.add_argument("--anchor-scale", type=float, default=0.95, help="Scale factor for pseudo-anchor vote weights.")
    parser.add_argument("--foreground-threshold", type=float, default=0.42, help="Fraction of total weight needed for foreground.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit processed videos; 0 means all.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip enhanced videos with complete final masks.")
    parser.add_argument("--disable-anchor-rerun", action="store_true", help="Only run diagnostics/fusion from baseline masks.")
    parser.add_argument("--require-anchor-rerun", action="store_true", help="Fail instead of falling back when no pseudo-anchor rerun completes.")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable conservative object mask postprocess.")
    parser.add_argument(
        "--save-overlays",
        default="sample",
        choices=["none", "sample", "all"],
        help="Write final debug overlay JPGs. 'sample' writes every --overlay-stride frames.",
    )
    parser.add_argument("--overlay-stride", type=int, default=12, help="Frame stride used when --save-overlays=sample.")
    parser.add_argument(
        "--save-anchor-overlays",
        default="none",
        choices=["none", "sample", "all"],
        help="Write overlay JPGs inside anchor_runs. Defaults to none to keep experiments small.",
    )
    parser.add_argument("--make-submission", action="store_true", help="Create and validate submission.zip after enhancement.")
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_format_spec(path: str | Path) -> FormatSpec:
    """Load a FormatSpec from JSON."""

    return FormatSpec.from_dict(_load_json(path))


def _prepare_data_info(args: argparse.Namespace, exp_dir: Path) -> dict[str, Any]:
    """Load or inspect data info and save it under the new experiment."""

    baseline_path = Path(args.baseline_exp).expanduser().resolve() / "data_info.json"
    if args.data_info:
        info = _load_json(args.data_info)
    elif baseline_path.exists():
        info = _load_json(baseline_path)
    else:
        data_info = inspect_dataset(Path(args.data_root).expanduser().resolve())
        save_data_info(data_info, exp_dir / "data_info.json")
        return data_info.to_dict()
    _atomic_write_json(exp_dir / "data_info.json", info)
    return info


def _prepare_format_spec(args: argparse.Namespace, data_info: dict[str, Any], exp_dir: Path) -> FormatSpec:
    """Load, inspect, or infer submission format."""

    baseline_path = Path(args.baseline_exp).expanduser().resolve() / "format_spec.json"
    if args.format_spec:
        spec = _load_format_spec(args.format_spec)
    elif baseline_path.exists():
        spec = _load_format_spec(baseline_path)
    elif args.sample_submission and Path(args.sample_submission).exists():
        spec = inspect_sample_submission(args.sample_submission)
    else:
        spec = infer_provisional_format(data_info)
    save_format_spec(spec, exp_dir / "format_spec.json")
    return spec


def _expected_masks_by_video(format_spec: FormatSpec) -> dict[str, list[dict[str, Any]]]:
    """Group expected masks by video id."""

    grouped: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in format_spec.expected_videos}
    for mask in format_spec.expected_masks:
        grouped.setdefault(str(mask.get("video_id")), []).append(dict(mask))
    for masks in grouped.values():
        masks.sort(key=lambda item: (int(item.get("frame_index", 0)), str(item.get("frame_stem", ""))))
    return grouped


def _video_by_id(data_info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return data-info videos keyed by id."""

    return {str(video.get("video_id")): dict(video) for video in data_info.get("videos", [])}


def _select_video_ids(format_spec: FormatSpec, max_videos: int) -> list[str]:
    """Select videos in submission order."""

    videos = list(format_spec.expected_videos) or sorted({str(mask.get("video_id")) for mask in format_spec.expected_masks})
    return videos[:max_videos] if max_videos else videos


def _subset_format_spec(format_spec: FormatSpec, video_ids: list[str], reason: str) -> FormatSpec:
    """Return a copy of a format spec restricted to selected videos."""

    keep = set(video_ids)
    return replace(
        format_spec,
        expected_videos=[video_id for video_id in format_spec.expected_videos if video_id in keep],
        expected_frame_count_per_video={
            video_id: count
            for video_id, count in format_spec.expected_frame_count_per_video.items()
            if video_id in keep
        },
        image_size_per_video={
            video_id: size
            for video_id, size in format_spec.image_size_per_video.items()
            if video_id in keep
        },
        expected_masks=[mask for mask in format_spec.expected_masks if str(mask.get("video_id")) in keep],
        sample_relative_paths=[
            path for path in format_spec.sample_relative_paths if any(f"/{video_id}/" in f"/{path}" for video_id in keep)
        ],
        notes=[*format_spec.notes, reason],
    )


def _baseline_masks_root(baseline_exp: str | Path) -> Path:
    """Resolve baseline masks root."""

    root = Path(baseline_exp).expanduser().resolve()
    return root / "masks" if (root / "masks").exists() else root


def _mask_path(root: Path, video_id: str, frame_stem: str) -> Path:
    """Return the expected indexed mask path under a mask root."""

    return root / video_id / f"{frame_stem}.png"


def _load_indexed_mask(path: str | Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an indexed mask as uint8, optionally resizing to expected (width, height)."""

    with Image.open(path) as image:
        if expected_size is not None and image.size != expected_size:
            image = image.resize(expected_size, Image.Resampling.NEAREST)
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.uint8)


def _save_indexed_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save an indexed mask as an 8-bit PNG."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), mode="L").save(output_path)


def _complete_existing_video(exp_dir: Path, masks: list[dict[str, Any]], video_id: str) -> bool:
    """Return whether final enhanced masks already exist with expected sizes."""

    if not masks:
        return False
    for mask in masks:
        path = exp_dir / "masks" / video_id / f"{mask['frame_stem']}.png"
        if not path.exists():
            return False
        width = int(mask.get("width") or 0)
        height = int(mask.get("height") or 0)
        if width and height:
            with Image.open(path) as image:
                if image.size != (width, height):
                    return False
    return True


def _object_ids_for_video(video: Mapping[str, Any], baseline_root: Path, video_id: str, masks: list[dict[str, Any]]) -> list[int]:
    """Collect object ids from prompts, then baseline first-frame mask."""

    ids: set[int] = set()
    for prompt in video.get("prompts", []) or []:
        for object_id in prompt.get("object_ids", []) or []:
            value = int(object_id)
            if value > 0:
                ids.add(value)
    if ids:
        return sorted(ids)
    if masks:
        first = masks[0]
        path = _mask_path(baseline_root, video_id, str(first["frame_stem"]))
        if path.exists():
            array = _load_indexed_mask(path)
            positives = [int(value) for value in np.unique(array).tolist() if int(value) > 0]
            if set(positives).issubset({1, 255}) and positives:
                ids.add(1)
            else:
                ids.update(value for value in positives if 0 < value <= 255)
    return sorted(ids) if ids else [1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a float."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _parse_anchor_fractions(value: str) -> list[float]:
    """Parse comma-separated anchor fractions."""

    fractions: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fraction = min(0.95, max(0.05, _safe_float(item, 0.5)))
        if fraction not in fractions:
            fractions.append(fraction)
    return fractions or [0.5]


def _mask_iou_bool(a: np.ndarray, b: np.ndarray) -> float:
    """Return IoU between two boolean masks."""

    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b.astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _bbox_center(box: list[int] | list[float] | None) -> tuple[float, float] | None:
    """Return bbox center."""

    if box is None:
        return None
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _build_diagnostics(
    video_id: str,
    masks: list[dict[str, Any]],
    object_ids: list[int],
    baseline_root: Path,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Compute self-supervised per-object diagnostics and frame quality."""

    rows: list[dict[str, Any]] = []
    quality: list[float] = []
    previous_by_object: dict[int, np.ndarray] = {}
    previous_area_by_object: dict[int, int] = {}
    previous_bbox_by_object: dict[int, list[int] | None] = {}

    for index, mask_info in enumerate(masks):
        frame_stem = str(mask_info["frame_stem"])
        width = int(mask_info.get("width") or 0)
        height = int(mask_info.get("height") or 0)
        expected_size = (width, height) if width and height else None
        path = _mask_path(baseline_root, video_id, frame_stem)
        indexed = _load_indexed_mask(path, expected_size)
        object_scores: list[float] = []
        for object_id in object_ids:
            current = indexed == int(object_id)
            area = int(current.sum())
            bbox = mask_to_bbox(current)
            prev = previous_by_object.get(object_id)
            prev_area = previous_area_by_object.get(object_id, area)
            prev_bbox = previous_bbox_by_object.get(object_id)
            temporal_iou = _mask_iou_bool(current, prev) if prev is not None else 1.0
            area_jump = abs(math.log((area + 1e-6) / (prev_area + 1e-6))) if index > 0 else 0.0
            area_stability = math.exp(-area_jump)
            present = 1.0 if area >= 4 else 0.0
            motion_iou = bbox_iou(bbox, prev_bbox) if bbox is not None and prev_bbox is not None else (1.0 if index == 0 else 0.0)
            center_jump = 0.0
            center = _bbox_center(bbox)
            prev_center = _bbox_center(prev_bbox)
            if center is not None and prev_center is not None:
                center_jump = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            score = 0.42 * temporal_iou + 0.28 * area_stability + 0.20 * present + 0.10 * motion_iou
            object_scores.append(float(np.clip(score, 0.0, 1.0)))
            rows.append(
                {
                    "video_id": video_id,
                    "frame_index": int(mask_info.get("frame_index", index)),
                    "frame_stem": frame_stem,
                    "object_id": int(object_id),
                    "area": area,
                    "bbox": json.dumps(bbox, ensure_ascii=True),
                    "temporal_iou": temporal_iou,
                    "area_jump": area_jump,
                    "area_stability": area_stability,
                    "motion_iou": motion_iou,
                    "center_jump": center_jump,
                    "self_quality": object_scores[-1],
                }
            )
            previous_by_object[object_id] = current
            previous_area_by_object[object_id] = area
            previous_bbox_by_object[object_id] = bbox
        if int(np.count_nonzero(indexed)) < 4:
            frame_quality = 0.0
        elif object_scores:
            frame_quality = float(np.mean(object_scores) * (0.7 + 0.3 * min(object_scores)))
        else:
            frame_quality = 0.0
        quality.append(frame_quality)
    return rows, quality


def _select_anchors(
    video_id: str,
    masks: list[dict[str, Any]],
    quality: list[float],
    baseline_root: Path,
    fractions: list[float],
    search_radius: int,
    max_anchor_runs: int,
    threshold: float,
) -> list[AnchorCandidate]:
    """Select pseudo anchors from baseline stability scores."""

    if len(masks) <= 2 or max_anchor_runs <= 0:
        return []
    selected: list[AnchorCandidate] = []
    used_indices: set[int] = set()
    for fraction in fractions:
        if len(selected) >= max_anchor_runs:
            break
        center = int(round(fraction * (len(masks) - 1)))
        start = max(1, center - max(0, search_radius))
        end = min(len(masks) - 2, center + max(0, search_radius))
        if start > end:
            continue
        def valid_candidates(indices: Iterable[int], require_threshold: bool = False) -> list[int]:
            valid: list[int] = []
            for idx in indices:
                if idx in used_indices:
                    continue
                score = float(quality[idx]) if idx < len(quality) else 0.0
                if require_threshold and score < threshold:
                    continue
                mask_info = masks[idx]
                frame_stem = str(mask_info["frame_stem"])
                mask_path = _mask_path(baseline_root, video_id, frame_stem)
                if not mask_path.exists():
                    continue
                indexed = _load_indexed_mask(mask_path)
                if int(np.count_nonzero(indexed)) < 4:
                    continue
                valid.append(idx)
            return valid

        candidates = valid_candidates(range(start, end + 1))
        if candidates:
            best_index = max(candidates, key=lambda idx: quality[idx] if idx < len(quality) else 0.0)
            best_quality = float(quality[best_index]) if best_index < len(quality) else 0.0
            if best_quality < threshold:
                candidates = []
        if not candidates:
            candidates = valid_candidates(range(1, len(masks) - 1), require_threshold=True)
            if candidates:
                best_index = min(
                    candidates,
                    key=lambda idx: (abs(idx - center), -(quality[idx] if idx < len(quality) else 0.0)),
                )
                best_quality = float(quality[best_index]) if best_index < len(quality) else 0.0
        if not candidates:
            continue
        mask_info = masks[best_index]
        frame_stem = str(mask_info["frame_stem"])
        mask_path = _mask_path(baseline_root, video_id, frame_stem)
        if not mask_path.exists():
            continue
        selected.append(
            AnchorCandidate(
                video_id=video_id,
                frame_index=best_index,
                frame_stem=frame_stem,
                quality=best_quality,
                source_mask_path=str(mask_path),
            )
        )
        for offset in range(best_index - search_radius, best_index + search_radius + 1):
            used_indices.add(offset)
    return selected


def _frame_abs_path(data_root: Path, frame: Mapping[str, Any]) -> str:
    """Resolve a frame path to an absolute path."""

    rel = str(frame.get("relative_path", ""))
    path = Path(rel)
    if not path.is_absolute():
        path = data_root / rel
    return str(path.resolve())


def _frames_for_expected_masks(video: Mapping[str, Any], masks: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    """Return frame metadata aligned with expected masks."""

    frames = list(video.get("frames", []) or [])
    if not frames:
        raise ValueError(f"{video.get('video_id')}: data_info has no frame list")
    frames_by_stem = {str(frame.get("frame_stem")): frame for frame in frames}
    aligned: list[dict[str, Any]] = []
    for order, mask_info in enumerate(masks):
        stem = str(mask_info["frame_stem"])
        frame = frames_by_stem.get(stem)
        if frame is None:
            if order >= len(frames):
                raise ValueError(f"{video.get('video_id')}: no data frame for expected mask {stem}")
            frame = frames[order]
        aligned.append(
            {
                "video_id": str(video.get("video_id")),
                "frame_index": order,
                "frame_stem": stem,
                "relative_path": _frame_abs_path(data_root, frame),
                "width": int(mask_info.get("width") or frame.get("width") or 0),
                "height": int(mask_info.get("height") or frame.get("height") or 0),
                "source_type": "image",
            }
        )
    return aligned


def _run_anchor(
    anchor: AnchorCandidate,
    video: Mapping[str, Any],
    masks: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    object_ids: list[int],
    exp_dir: Path,
    data_root: Path,
    predictor: Any,
    args: argparse.Namespace,
) -> AnchorCandidate:
    """Run SAM2 from one pseudo-anchor frame to the end of the video."""

    video_id = str(video.get("video_id"))
    run_dir = exp_dir / "anchor_runs" / video_id / f"anchor_{anchor.frame_stem}"
    anchor.run_dir = str(run_dir)
    sub_frames = frames[anchor.frame_index :]
    sub_stems = [str(mask["frame_stem"]) for mask in masks[anchor.frame_index :]]
    pseudo_video = {
        "video_id": video_id,
        "source_type": "frames",
        "relative_path": str(Path(sub_frames[0]["relative_path"]).parent),
        "frame_count": len(sub_frames),
        "width": sub_frames[0].get("width"),
        "height": sub_frames[0].get("height"),
        "frames": sub_frames,
        "prompts": [],
    }
    pseudo_prompts = [
        {
            "video_id": video_id,
            "prompt_type": "mask",
            "relative_path": str(Path(anchor.source_mask_path).resolve()),
            "frame_index": 0,
            "object_ids": object_ids,
        }
    ]
    config = {
        "data_root": str(data_root),
        "cache_dir": str(exp_dir / "cache" / "anchor_frames" / video_id / f"anchor_{anchor.frame_stem}"),
        "predictor": predictor,
        "device": "cuda",
        "prompt_mode": args.prompt_mode,
        "resize_long_side": int(args.resize_long_side or 0),
        "save_raw_logits": False,
        "save_overlays": args.save_anchor_overlays,
        "overlay_stride": int(args.overlay_stride or 12),
        "output_frame_stems": sub_stems,
        "box_padding_ratio": 0.08,
        "num_positive": 5,
        "num_negative": 8,
        "negative_r1": 8,
        "negative_r2": 25,
    }
    result = run_sam2_on_video(pseudo_video, pseudo_prompts, run_dir, config)
    anchor.warnings.extend(result.warnings)
    if result.status == "done":
        anchor.status = "done"
    else:
        anchor.status = "failed"
        anchor.error = result.error
    return anchor


def _contribution_weight(kind: str, frame_index: int, anchor: AnchorCandidate | None, args: argparse.Namespace) -> float:
    """Return a temporal vote weight."""

    tau = max(1e-6, float(args.fusion_tau))
    if kind == "baseline":
        return float(args.baseline_floor) + (1.0 - float(args.baseline_floor)) * math.exp(-max(0, frame_index) / tau)
    if anchor is None:
        return 0.0
    distance = max(0, frame_index - anchor.frame_index)
    return float(args.anchor_scale) * float(anchor.quality) * math.exp(-distance / tau)


def _weighted_vote_indexed(
    video_id: str,
    mask_info: Mapping[str, Any],
    object_ids: list[int],
    baseline_root: Path,
    anchors: list[AnchorCandidate],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse baseline and anchor indexed masks by weighted object votes."""

    frame_index = int(mask_info.get("frame_index", 0))
    frame_stem = str(mask_info["frame_stem"])
    width = int(mask_info.get("width") or 0)
    height = int(mask_info.get("height") or 0)
    expected_size = (width, height) if width and height else None
    contributions: list[tuple[str, float, np.ndarray]] = []
    baseline_path = _mask_path(baseline_root, video_id, frame_stem)
    baseline = _load_indexed_mask(baseline_path, expected_size)
    contributions.append(("baseline", _contribution_weight("baseline", frame_index, None, args), baseline))

    for anchor in anchors:
        if anchor.status != "done" or anchor.run_dir is None or frame_index < anchor.frame_index:
            continue
        anchor_path = Path(anchor.run_dir) / "masks" / video_id / f"{frame_stem}.png"
        if not anchor_path.exists():
            continue
        contributions.append(
            (
                f"anchor_{anchor.frame_stem}",
                _contribution_weight("anchor", frame_index, anchor, args),
                _load_indexed_mask(anchor_path, expected_size or (baseline.shape[1], baseline.shape[0])),
            )
        )

    shape = baseline.shape[:2]
    if not object_ids:
        return baseline, {"mode": "baseline_no_object_ids", "contributors": []}
    score_planes = np.zeros((len(object_ids), shape[0], shape[1]), dtype=np.float32)
    total_weight = 0.0
    debug_contrib: list[dict[str, Any]] = []
    for name, weight, indexed in contributions:
        if indexed.shape[:2] != shape:
            indexed = np.asarray(Image.fromarray(indexed).resize((shape[1], shape[0]), Image.Resampling.NEAREST))
        total_weight += float(weight)
        debug_contrib.append({"name": name, "weight": float(weight), "foreground": int((indexed > 0).sum())})
        for object_index, object_id in enumerate(object_ids):
            score_planes[object_index][indexed == int(object_id)] += float(weight)

    winners = np.argmax(score_planes, axis=0)
    max_scores = np.max(score_planes, axis=0)
    threshold = float(args.foreground_threshold) * max(total_weight, 1e-6)
    output = np.zeros(shape, dtype=np.uint8)
    foreground = max_scores > threshold
    for object_index, object_id in enumerate(object_ids):
        output[foreground & (winners == object_index)] = np.uint8(min(max(int(object_id), 0), 255))
    return output, {
        "mode": "weighted_vote",
        "frame_stem": frame_stem,
        "frame_index": frame_index,
        "total_weight": float(total_weight),
        "threshold": float(threshold),
        "contributors": debug_contrib,
    }


def _target_type_from_initial(initial_mask: np.ndarray, object_id: int) -> str:
    """Return a conservative tiny/regular target type."""

    obj = initial_mask == int(object_id)
    area = int(obj.sum())
    if area <= 0:
        return "regular"
    bbox = mask_to_bbox(obj)
    min_side = 0 if bbox is None else min(int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1))
    ratio = area / float(max(1, obj.shape[0] * obj.shape[1]))
    return "tiny" if ratio < 0.003 or min_side < 24 else "regular"


def _postprocess_indexed(
    indexed: np.ndarray,
    object_ids: list[int],
    initial_mask: np.ndarray,
    recent_areas: dict[int, list[float]],
    enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply conservative per-object postprocess and merge without expanding IDs."""

    if not enabled:
        return indexed, {"enabled": False}
    output = np.zeros_like(indexed, dtype=np.uint8)
    debug: dict[str, Any] = {"enabled": True, "objects": {}}
    cfg = PostprocessConfig(
        tiny_min_area_abs=2,
        tiny_min_area_ratio=0.01,
        regular_min_area_abs=8,
        regular_min_area_ratio=0.001,
        close_kernel_size=3,
        open_kernel_size=1,
        smooth_kernel_size=1,
        keep_largest_component=False,
        area_min_recent_ratio=0.08,
        area_max_recent_ratio=8.0,
    )
    for object_id in object_ids:
        raw = indexed == int(object_id)
        target_type = _target_type_from_initial(initial_mask, int(object_id))
        prediction = ObjectMaskPrediction(
            object_id=int(object_id),
            mask=raw,
            reliability=0.5,
            area_initial=float(max(1, int((initial_mask == int(object_id)).sum()))),
            target_type=target_type,
        )
        try:
            result = postprocess_object_mask(prediction, recent_areas=recent_areas.get(int(object_id), []), config=cfg)
            use_mask = result.mask
            if result.abnormal_area and raw.sum() > 0:
                use_mask = raw
            output[use_mask & (output == 0)] = np.uint8(min(max(int(object_id), 0), 255))
            area = int(use_mask.sum())
            recent_areas.setdefault(int(object_id), []).append(float(area))
            recent_areas[int(object_id)] = recent_areas[int(object_id)][-8:]
            debug["objects"][str(object_id)] = {
                "target_type": target_type,
                "raw_area": int(raw.sum()),
                "final_area": area,
                "abnormal_area": bool(result.abnormal_area),
                "fragmented": bool(result.fragmented),
            }
        except Exception as exc:
            output[raw & (output == 0)] = np.uint8(min(max(int(object_id), 0), 255))
            debug["objects"][str(object_id)] = {"error": f"{type(exc).__name__}: {exc}", "raw_area": int(raw.sum())}
    return output, debug


def _save_overlay(frame_path: str | Path, mask: np.ndarray, output_path: Path) -> None:
    """Save a lightweight debug overlay."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
    if mask.shape[:2] != (rgb.height, rgb.width):
        mask = np.asarray(Image.fromarray(mask).resize(rgb.size, Image.Resampling.NEAREST))
    arr = np.asarray(rgb).copy()
    palette = [
        ImageColor.getrgb("#ff4040"),
        ImageColor.getrgb("#34c759"),
        ImageColor.getrgb("#0a84ff"),
        ImageColor.getrgb("#ffcc00"),
        ImageColor.getrgb("#bf5af2"),
        ImageColor.getrgb("#ff9f0a"),
    ]
    for index, object_id in enumerate(sorted(int(value) for value in np.unique(mask) if int(value) > 0)):
        region = mask == object_id
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        arr[region] = (0.55 * arr[region] + 0.45 * color).astype(np.uint8)
    Image.fromarray(arr).save(output_path, quality=90)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to CSV with a union field set."""

    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def _build_predictor(args: argparse.Namespace, exp_dir: Path) -> Any:
    """Build the SAM2 predictor for pseudo-anchor reruns."""

    repo_dir = Path(args.sam2_repo_dir).expanduser().resolve() if args.sam2_repo_dir else exp_dir / "external" / "sam2"
    install_or_check_sam2(repo_dir)
    checkpoint_candidate = Path(args.checkpoint).expanduser()
    if checkpoint_candidate.exists():
        checkpoint = checkpoint_candidate.resolve()
    else:
        checkpoint = download_sam2_checkpoint(args.checkpoint, exp_dir / "checkpoints")
    return build_sam2_video_predictor(
        checkpoint_path=checkpoint,
        model_cfg=args.model_cfg,
        device="cuda",
        vos_optimized=False,
    )


def _write_failure_sanity(exp_dir: Path, errors: list[str], warnings: list[str]) -> None:
    """Write a failed sanity_check.json."""

    _atomic_write_json(
        exp_dir / "sanity_check.json",
        {
            "passed": False,
            "errors": errors,
            "warnings": warnings,
            "submission_validation": "not_run",
        },
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    baseline_root = _baseline_masks_root(args.baseline_exp)
    outputs_root = Path(args.output_dir).expanduser().resolve()
    exp_dir = outputs_root / args.experiment_id
    logs_dir = exp_dir / "logs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = logs_dir / "per_video_status.json"

    errors: list[str] = []
    warnings: list[str] = []
    status: dict[str, Any] = {"videos": {}, "summary": {}}
    all_diagnostics: list[dict[str, Any]] = []
    all_fusion_rows: list[dict[str, Any]] = []
    anchor_run_attempts = 0
    anchor_run_successes = 0
    videos_with_done_anchors = 0

    if args.require_anchor_rerun and (args.disable_anchor_rerun or int(args.max_anchor_runs or 0) <= 0):
        errors.append("--require-anchor-rerun needs anchor reruns enabled and --max-anchor-runs > 0.")
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    try:
        data_info = _prepare_data_info(args, exp_dir)
        format_spec = _prepare_format_spec(args, data_info, exp_dir)
        videos_by_id = _video_by_id(data_info)
        selected_video_ids = _select_video_ids(format_spec, int(args.max_videos or 0))
        if int(args.max_videos or 0):
            format_spec = _subset_format_spec(format_spec, selected_video_ids, f"restricted_by_max_videos={int(args.max_videos)}")
            save_format_spec(format_spec, exp_dir / "format_spec.json")
        expected_by_video = _expected_masks_by_video(format_spec)
        anchor_fractions = _parse_anchor_fractions(args.anchor_fractions)
    except Exception as exc:
        errors.append(f"Setup failed: {type(exc).__name__}: {exc}")
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    predictor: Any | None = None
    if not args.disable_anchor_rerun and int(args.max_anchor_runs or 0) > 0:
        try:
            predictor = _build_predictor(args, exp_dir)
        except Exception as exc:
            message = f"SAM2 pseudo-anchor setup failed: {type(exc).__name__}: {exc}"
            if args.require_anchor_rerun:
                errors.append(message)
                status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
                _atomic_write_json(status_path, status)
                _write_failure_sanity(exp_dir, errors, warnings)
                return 1
            warnings.append(f"{message}; falling back to baseline-only fusion.")
            predictor = None

    for video_id in selected_video_ids:
        video = videos_by_id.get(video_id)
        masks = expected_by_video.get(video_id, [])
        if video is None or not masks:
            message = f"{video_id}: missing video metadata or expected masks"
            errors.append(message)
            status["videos"][video_id] = {"status": "failed", "error": message}
            _atomic_write_json(status_path, status)
            continue
        if args.skip_existing and _complete_existing_video(exp_dir, masks, video_id):
            result = VideoEnhanceResult(
                video_id=video_id,
                status="skipped_existing",
                frame_count=len(masks),
                object_ids=[],
                warnings=["--skip-existing and masks are complete"],
            )
            status["videos"][video_id] = result.to_dict()
            _atomic_write_json(status_path, status)
            continue

        try:
            object_ids = _object_ids_for_video(video, baseline_root, video_id, masks)
            diagnostics, quality = _build_diagnostics(video_id, masks, object_ids, baseline_root)
            all_diagnostics.extend(diagnostics)
            anchors = _select_anchors(
                video_id=video_id,
                masks=masks,
                quality=quality,
                baseline_root=baseline_root,
                fractions=anchor_fractions,
                search_radius=int(args.anchor_search_radius),
                max_anchor_runs=0 if predictor is None else int(args.max_anchor_runs),
                threshold=float(args.anchor_quality_threshold),
            )
            frames = _frames_for_expected_masks(video, masks, data_root)
            for anchor in anchors:
                _run_anchor(anchor, video, masks, frames, object_ids, exp_dir, data_root, predictor, args)
            anchor_run_attempts += len(anchors)
            done_anchors = sum(1 for anchor in anchors if anchor.status == "done")
            anchor_run_successes += done_anchors
            if done_anchors > 0:
                videos_with_done_anchors += 1

            initial_mask = _load_indexed_mask(_mask_path(baseline_root, video_id, str(masks[0]["frame_stem"])))
            recent_areas: dict[int, list[float]] = {
                object_id: [float((initial_mask == int(object_id)).sum())] for object_id in object_ids
            }
            output_paths: list[str] = []
            for frame_order, (mask_info, frame) in enumerate(zip(masks, frames)):
                if frame_order == 0:
                    final_mask = initial_mask.copy()
                    fusion_debug = {
                        "mode": "preserve_baseline_first_frame",
                        "frame_stem": str(mask_info["frame_stem"]),
                        "frame_index": int(mask_info.get("frame_index", 0)),
                    }
                    post_debug = {"enabled": False, "reason": "preserve_baseline_first_frame"}
                else:
                    fused, fusion_debug = _weighted_vote_indexed(video_id, mask_info, object_ids, baseline_root, anchors, args)
                    final_mask, post_debug = _postprocess_indexed(
                        fused,
                        object_ids,
                        initial_mask,
                        recent_areas,
                        enabled=not bool(args.disable_postprocess),
                    )
                frame_stem = str(mask_info["frame_stem"])
                mask_path = exp_dir / "masks" / video_id / f"{frame_stem}.png"
                _save_indexed_mask(final_mask, mask_path)
                output_paths.append(str(mask_path))
                frame_index = int(mask_info.get("frame_index", len(output_paths) - 1))
                save_overlay = (
                    args.save_overlays == "all"
                    or (
                        args.save_overlays == "sample"
                        and frame_index % max(1, int(args.overlay_stride or 12)) == 0
                    )
                )
                if save_overlay:
                    _save_overlay(frame["relative_path"], final_mask, exp_dir / "overlays" / video_id / f"{frame_stem}.jpg")
                all_fusion_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "frame_stem": frame_stem,
                        "object_ids": json.dumps(object_ids, ensure_ascii=True),
                        "fusion_debug": json.dumps(fusion_debug, ensure_ascii=True),
                        "postprocess_debug": json.dumps(post_debug, ensure_ascii=True),
                    }
                )

            video_warnings = [warning for anchor in anchors for warning in anchor.warnings]
            video_warnings.extend(
                f"anchor {anchor.frame_stem} failed: {anchor.error}"
                for anchor in anchors
                if anchor.status == "failed"
            )
            result = VideoEnhanceResult(
                video_id=video_id,
                status="done",
                frame_count=len(masks),
                object_ids=object_ids,
                anchors=anchors,
                mask_paths=output_paths,
                warnings=video_warnings,
            )
            warnings.extend(f"{video_id}: {warning}" for warning in video_warnings)
            status["videos"][video_id] = result.to_dict()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"{video_id}: {message}")
            status["videos"][video_id] = VideoEnhanceResult(
                video_id=video_id,
                status="failed",
                frame_count=len(masks),
                object_ids=[],
                error=message,
            ).to_dict()
        _atomic_write_json(status_path, status)

    _write_csv(logs_dir / "self_diagnostics.csv", all_diagnostics)
    _write_csv(logs_dir / "fusion_debug.csv", all_fusion_rows)

    if args.require_anchor_rerun and anchor_run_successes <= 0:
        errors.append("Required pseudo-anchor rerun, but no anchor runs completed successfully.")

    if errors:
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    if args.make_submission:
        try:
            submission_path = exp_dir / "submission.zip"
            make_submission(exp_dir / "masks", submission_path, format_spec)
            sanity = validate_submission_zip(submission_path, format_spec, data_info)
            _atomic_write_json(exp_dir / "sanity_check.json", sanity)
            if not sanity["passed"]:
                status["summary"] = {"status": "failed_validation", "errors": sanity["errors"], "warnings": sanity["warnings"]}
                _atomic_write_json(status_path, status)
                return 1
        except Exception as exc:
            errors.append(f"Submission creation/validation failed: {type(exc).__name__}: {exc}")
            status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
            _atomic_write_json(status_path, status)
            _write_failure_sanity(exp_dir, errors, warnings)
            return 1

    status["summary"] = {
        "status": "done",
        "num_videos": len(selected_video_ids),
        "baseline_exp": str(Path(args.baseline_exp).expanduser().resolve()),
        "anchor_run_attempts": anchor_run_attempts,
        "anchor_run_successes": anchor_run_successes,
        "videos_with_done_anchors": videos_with_done_anchors,
        "warnings": warnings,
        "submission_zip": str(exp_dir / "submission.zip") if args.make_submission else None,
        "diagnostics_csv": str(logs_dir / "self_diagnostics.csv"),
        "fusion_debug_csv": str(logs_dir / "fusion_debug.csv"),
    }
    _atomic_write_json(status_path, status)
    print(json.dumps(status["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

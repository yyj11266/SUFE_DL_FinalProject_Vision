"""Fuse SAM2 baseline masks with a Cutie candidate using conservative object gates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspect_sufe import DataInfo, VideoInfo, inspect_dataset, save_data_info
from src.data.submission import (
    FormatSpec,
    infer_provisional_format,
    inspect_sample_submission,
    make_submission,
    save_format_spec,
    validate_submission_zip,
)
from src.trackers.sam2_tracker import _load_mask_prompt, _prepare_frame_dir, _resize_mask, _save_indexed_mask
from src.vos.conservative_fusion import ConservativeFusionConfig, fuse_frame, object_ids_from_indexed


SMOKE_VIDEO_IDS = ["0u8fy7u2", "2b827e3a", "2a1jkxdf", "kpg9gld7", "lkob5diu", "pjlde9hu"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse SAM2 baseline and Cutie candidate masks.")
    parser.add_argument("--data-root", required=True, help="Extracted SUFE data root.")
    parser.add_argument("--baseline-exp", required=True, help="SAM2 baseline experiment directory.")
    parser.add_argument("--cutie-exp", required=True, help="Cutie candidate experiment directory.")
    parser.add_argument("--sample-submission", help="Optional sample_submission.zip path.")
    parser.add_argument("--output-dir", required=True, help="Outputs root directory.")
    parser.add_argument("--experiment-id", default=f"sam2_cutie_fusion_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--video-ids", default="", help="Comma-separated video IDs. Overrides --max-videos ordering when set.")
    parser.add_argument("--smoke", action="store_true", help="Run the fixed six-video smoke subset.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-cutie-area", type=int, default=16)
    parser.add_argument("--min-sam2-iou", type=float, default=0.50)
    parser.add_argument("--min-temporal-iou", type=float, default=0.35)
    parser.add_argument("--min-area-ratio", type=float, default=0.20)
    parser.add_argument("--max-area-ratio", type=float, default=3.50)
    parser.add_argument("--disallow-cutie-when-sam2-empty", action="store_true")
    parser.add_argument("--make-submission", action="store_true")
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _load_format_spec(path: Path) -> FormatSpec:
    return FormatSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _video_lookup(data_info: DataInfo) -> dict[str, VideoInfo]:
    return {video.video_id: video for video in data_info.videos}


def _expected_masks_for_video(format_spec: FormatSpec, video_id: str) -> list[dict[str, Any]]:
    return sorted(
        [mask for mask in format_spec.expected_masks if str(mask.get("video_id")) == str(video_id)],
        key=lambda item: int(item.get("frame_index", 0)),
    )


def _prepare_format_spec(
    data_info: DataInfo,
    sample_submission: str | None,
    baseline_exp: Path,
    cutie_exp: Path,
    output_path: Path,
) -> FormatSpec:
    if sample_submission and Path(sample_submission).exists():
        spec = inspect_sample_submission(sample_submission)
    elif (baseline_exp / "format_spec.json").exists():
        spec = _load_format_spec(baseline_exp / "format_spec.json")
    elif (cutie_exp / "format_spec.json").exists():
        spec = _load_format_spec(cutie_exp / "format_spec.json")
    else:
        spec = infer_provisional_format(data_info.to_dict())
    save_format_spec(spec, output_path)
    return spec


def _subset_format_spec(format_spec: FormatSpec, video_ids: list[str], max_frames: int, reason: str) -> FormatSpec:
    keep = set(video_ids)
    expected_masks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for video_id in video_ids:
        masks = [mask for mask in format_spec.expected_masks if str(mask.get("video_id")) == video_id]
        if max_frames > 0:
            masks = sorted(masks, key=lambda item: int(item.get("frame_index", 0)))[:max_frames]
        expected_masks.extend(masks)
        counts[video_id] = len(masks) if masks else min(max_frames, format_spec.expected_frame_count_per_video.get(video_id, 0))
    return FormatSpec(
        relative_mask_path_pattern=format_spec.relative_mask_path_pattern,
        mask_extension=format_spec.mask_extension,
        mask_encoding=format_spec.mask_encoding,
        object_id_encoding_rule=format_spec.object_id_encoding_rule,
        expected_videos=[video_id for video_id in format_spec.expected_videos if video_id in keep],
        expected_frame_count_per_video=counts,
        image_size_per_video={
            video_id: size for video_id, size in format_spec.image_size_per_video.items() if video_id in keep
        },
        not_verified_by_sample=format_spec.not_verified_by_sample,
        expected_masks=expected_masks,
        sample_relative_paths=[str(mask["relative_path"]) for mask in expected_masks if mask.get("relative_path")],
        format_source=format_spec.format_source,
        notes=[*format_spec.notes, reason],
    )


def _selected_videos(data_info: DataInfo, format_spec: FormatSpec, args: argparse.Namespace) -> tuple[list[VideoInfo], list[str]]:
    videos = _video_lookup(data_info)
    missing: list[str] = []
    if args.smoke:
        requested = SMOKE_VIDEO_IDS
    elif args.video_ids.strip():
        requested = [part.strip() for part in args.video_ids.split(",") if part.strip()]
    else:
        requested = list(format_spec.expected_videos) or [video.video_id for video in data_info.videos]
    selected: list[VideoInfo] = []
    for video_id in requested:
        video = videos.get(video_id)
        if video is None:
            missing.append(video_id)
            continue
        selected.append(video)
        if args.max_videos and not (args.video_ids.strip() or args.smoke) and len(selected) >= args.max_videos:
            break
    return selected, missing


def _read_indexed_mask(exp_dir: Path, mask: dict[str, Any], source_label: str) -> np.ndarray:
    video_id = str(mask.get("video_id"))
    frame_stem = str(mask.get("frame_stem"))
    relative_path = str(mask.get("relative_path") or "")
    candidates = [
        exp_dir / "masks" / video_id / f"{frame_stem}.png",
        exp_dir / video_id / f"{frame_stem}.png",
        exp_dir / "Annotations" / video_id / f"{frame_stem}.png",
    ]
    if relative_path:
        rel = Path(PurePosixPath(relative_path).as_posix())
        candidates.extend([exp_dir / rel, exp_dir / "masks" / rel])
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(f"{source_label} mask missing for {video_id}/{frame_stem}; checked {candidates[:5]}")
    with Image.open(source) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] >= 3 and np.array_equal(array[..., 0], array[..., 1]) and np.array_equal(array[..., 0], array[..., 2]):
            array = array[..., 0]
        else:
            array = np.any(array[..., :3] > 0, axis=-1).astype(np.uint8)
    return array.astype(np.uint8)


def _resize_to_reference(mask: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if mask.shape[:2] == reference.shape[:2]:
        return mask.astype(np.uint8, copy=False)
    height, width = reference.shape[:2]
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)).astype(np.uint8)


def _initial_prompt(video: VideoInfo, data_root: Path, exp_dir: Path) -> np.ndarray:
    frames = _prepare_frame_dir(
        video,
        {
            "data_root": str(data_root),
            "cache_dir": str(exp_dir / "cache"),
            "resize_long_side": 0,
        },
    )
    if not frames:
        raise RuntimeError(f"{video.video_id}: no frames found")
    first = frames[0]
    initial = _load_mask_prompt(video.prompts, data_root)
    if initial.ndim == 3:
        initial = np.any(initial[..., :3] > 0, axis=-1).astype(np.uint8)
    return _resize_mask(initial, first.original_width, first.original_height).astype(np.uint8)


def _complete_existing_video(exp_dir: Path, format_spec: FormatSpec, video: VideoInfo) -> bool:
    masks = _expected_masks_for_video(format_spec, video.video_id)
    if not masks:
        return False
    for mask in masks:
        path = exp_dir / "masks" / video.video_id / f"{mask['frame_stem']}.png"
        if not path.exists():
            return False
        expected_width = int(mask.get("width") or video.width or 0)
        expected_height = int(mask.get("height") or video.height or 0)
        if expected_width and expected_height:
            with Image.open(path) as image:
                if image.size != (expected_width, expected_height):
                    return False
    return True


def _write_failure_sanity(exp_dir: Path, errors: list[str], warnings: list[str]) -> None:
    _atomic_json(
        exp_dir / "sanity_check.json",
        {"passed": False, "errors": errors, "warnings": warnings, "submission_validation": "not_run"},
    )


def _fusion_config(args: argparse.Namespace) -> ConservativeFusionConfig:
    return ConservativeFusionConfig(
        min_cutie_area=int(args.min_cutie_area),
        min_sam2_iou=float(args.min_sam2_iou),
        min_temporal_iou=float(args.min_temporal_iou),
        min_area_ratio=float(args.min_area_ratio),
        max_area_ratio=float(args.max_area_ratio),
        allow_cutie_when_sam2_empty=not bool(args.disallow_cutie_when_sam2_empty),
    )


def _run_video(
    video: VideoInfo,
    data_root: Path,
    baseline_exp: Path,
    cutie_exp: Path,
    exp_dir: Path,
    format_spec: FormatSpec,
    config: ConservativeFusionConfig,
    debug_writer: csv.DictWriter[str],
) -> dict[str, Any]:
    first_frame = _initial_prompt(video, data_root, exp_dir)
    object_ids = object_ids_from_indexed(first_frame)
    if not object_ids:
        raise RuntimeError(f"{video.video_id}: first-frame prompt has no positive object ids")
    expected_masks = _expected_masks_for_video(format_spec, video.video_id)
    if not expected_masks:
        expected_masks = [
            {
                "video_id": video.video_id,
                "frame_stem": frame.frame_stem,
                "frame_index": frame.frame_index,
                "width": frame.width,
                "height": frame.height,
                "relative_path": f"{video.video_id}/{frame.frame_stem}.png",
            }
            for frame in video.frames
        ]
    warnings: list[str] = []
    previous_output: np.ndarray | None = None
    source_counts = {"prompt": 0, "sam2": 0, "cutie": 0}
    missing_output_frames: dict[str, list[int]] = {str(object_id): [] for object_id in object_ids}

    for order, mask_meta in enumerate(expected_masks):
        frame_index = int(mask_meta.get("frame_index", order))
        frame_stem = str(mask_meta.get("frame_stem"))
        sam2 = _resize_to_reference(_read_indexed_mask(baseline_exp, mask_meta, "SAM2"), first_frame)
        cutie = _resize_to_reference(_read_indexed_mask(cutie_exp, mask_meta, "Cutie"), first_frame)
        result = fuse_frame(
            sam2_indexed=sam2,
            cutie_indexed=cutie,
            object_ids=object_ids,
            frame_index=frame_index,
            first_frame_mask=first_frame if frame_index == 0 else None,
            previous_output=previous_output,
            config=config,
        )
        output = result.indexed_mask.astype(np.uint8, copy=False)
        output_path = exp_dir / "masks" / video.video_id / f"{frame_stem}.png"
        _save_indexed_mask(output, output_path)
        previous_output = output
        warnings.extend(result.warnings)
        present = set(object_ids_from_indexed(output))
        for object_id in object_ids:
            if object_id not in present:
                missing_output_frames[str(object_id)].append(frame_index)
        for decision in result.decisions:
            source_counts[decision.source] = source_counts.get(decision.source, 0) + 1
            row = decision.to_dict()
            debug_writer.writerow(
                {
                    **row,
                    "video_id": video.video_id,
                    "frame_index": frame_index,
                    "frame_stem": frame_stem,
                    "frame_warnings": ";".join(result.warnings),
                }
            )

    first_path = exp_dir / "masks" / video.video_id / f"{expected_masks[0]['frame_stem']}.png"
    with Image.open(first_path) as image:
        first_written = np.asarray(image)
    return {
        "status": "done",
        "frame_count": len(expected_masks),
        "object_ids": object_ids,
        "first_frame_exact": bool(np.array_equal(first_written, first_frame)),
        "source_counts": source_counts,
        "missing_output_frames": missing_output_frames,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    data_root = Path(args.data_root).expanduser().resolve()
    baseline_exp = Path(args.baseline_exp).expanduser().resolve()
    cutie_exp = Path(args.cutie_exp).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    exp_dir = output_root / args.experiment_id
    logs_dir = exp_dir / "logs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_info = inspect_dataset(data_root)
    save_data_info(data_info, exp_dir / "data_info.json")
    format_spec = _prepare_format_spec(data_info, args.sample_submission, baseline_exp, cutie_exp, exp_dir / "format_spec.json")
    selected, missing = _selected_videos(data_info, format_spec, args)
    if selected and (args.smoke or args.video_ids.strip() or args.max_videos or args.max_frames):
        format_spec = _subset_format_spec(
            format_spec,
            [video.video_id for video in selected],
            int(args.max_frames),
            "restricted_by_sam2_cutie_fusion_selection",
        )
        save_format_spec(format_spec, exp_dir / "format_spec.json")

    status: dict[str, Any] = {"videos": {}, "summary": {}}
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        errors.append(f"Requested videos not found: {missing}")
    if not baseline_exp.exists():
        errors.append(f"SAM2 baseline experiment directory does not exist: {baseline_exp}")
    if not cutie_exp.exists():
        errors.append(f"Cutie experiment directory does not exist: {cutie_exp}")
    config = _fusion_config(args)

    fieldnames = [
        "video_id",
        "frame_index",
        "frame_stem",
        "object_id",
        "source",
        "reason",
        "sam2_area",
        "cutie_area",
        "output_area",
        "sam2_cutie_iou",
        "cutie_temporal_iou",
        "cutie_area_ratio",
        "warnings",
        "frame_warnings",
    ]
    with (logs_dir / "fusion_debug.csv").open("w", newline="", encoding="utf-8") as handle:
        writer: csv.DictWriter[str] = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video in selected:
            if errors:
                break
            try:
                if args.skip_existing and _complete_existing_video(exp_dir, format_spec, video):
                    status["videos"][video.video_id] = {
                        "status": "skipped_existing",
                        "frame_count": len(_expected_masks_for_video(format_spec, video.video_id)),
                        "reason": "--skip-existing and masks are complete",
                    }
                    _atomic_json(logs_dir / "per_video_status.json", status)
                    continue
                video_status = _run_video(
                    video=video,
                    data_root=data_root,
                    baseline_exp=baseline_exp,
                    cutie_exp=cutie_exp,
                    exp_dir=exp_dir,
                    format_spec=format_spec,
                    config=config,
                    debug_writer=writer,
                )
                status["videos"][video.video_id] = video_status
                if not video_status.get("first_frame_exact"):
                    errors.append(f"{video.video_id}: first frame is not exact")
                warnings.extend(f"{video.video_id}: {warning}" for warning in video_status.get("warnings", []))
            except Exception as exc:
                errors.append(f"{video.video_id}: {type(exc).__name__}: {exc}")
                status["videos"][video.video_id] = {"status": "failed", "error": errors[-1]}
            _atomic_json(logs_dir / "per_video_status.json", status)

    if errors:
        status["summary"] = {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "fusion_config": asdict(config),
        }
        _atomic_json(logs_dir / "per_video_status.json", status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    if args.make_submission:
        try:
            submission_path = exp_dir / "submission.zip"
            make_submission(exp_dir / "masks", submission_path, format_spec)
            sanity = validate_submission_zip(submission_path, format_spec, data_info.to_dict())
            _atomic_json(exp_dir / "sanity_check.json", sanity)
            if not sanity["passed"]:
                status["summary"] = {
                    "status": "failed_validation",
                    "errors": sanity["errors"],
                    "warnings": sanity["warnings"],
                }
                _atomic_json(logs_dir / "per_video_status.json", status)
                return 1
        except Exception as exc:
            errors.append(f"Submission creation/validation failed: {type(exc).__name__}: {exc}")
            status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
            _atomic_json(logs_dir / "per_video_status.json", status)
            _write_failure_sanity(exp_dir, errors, warnings)
            return 1
    else:
        _atomic_json(
            exp_dir / "sanity_check.json",
            {
                "passed": True,
                "submission_validation": "not_run",
                "candidate_only": True,
                "num_videos": len(selected),
                "warnings": warnings,
            },
        )

    manifest = {
        "experiment_id": args.experiment_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": "sam2_cutie_conservative_fusion",
        "baseline_exp": str(baseline_exp),
        "cutie_exp": str(cutie_exp),
        "selected_videos": [video.video_id for video in selected],
        "first_frame_policy": "copy_input_annotation_exactly",
        "fallback_policy": "sam2_current_frame_anchor_no_previous_recovery",
        "fusion_config": asdict(config),
    }
    _atomic_json(exp_dir / "run_manifest.json", manifest)
    status["summary"] = {
        "status": "done",
        "num_videos": len(selected),
        "warnings": warnings,
        "submission_zip": str(exp_dir / "submission.zip") if args.make_submission else None,
    }
    _atomic_json(logs_dir / "per_video_status.json", status)
    print(json.dumps(status["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

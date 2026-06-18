"""Run the SAM2.1 Hiera Large baseline for SUFE VOS leaderboard data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from src.trackers.sam2_tracker import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_MODEL_CFG,
    build_sam2_video_predictor,
    download_sam2_checkpoint,
    install_or_check_sam2,
    run_sam2_on_video,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the SAM2 baseline runner."""

    parser = argparse.ArgumentParser(description="Run SAM2.1 Hiera Large baseline on SUFE VOS data.")
    parser.add_argument("--data-root", required=True, help="Extracted SUFE data root.")
    parser.add_argument("--sample-submission", help="Optional sample_submission.zip path.")
    parser.add_argument("--output-dir", required=True, help="Outputs root directory.")
    parser.add_argument("--experiment-id", default=f"sam2_baseline_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_NAME, help="Checkpoint path or known checkpoint name.")
    parser.add_argument("--sam2-repo-dir", default=None, help="Reusable official SAM2 repo directory; prefer Colab local disk.")
    parser.add_argument("--model-cfg", default=DEFAULT_MODEL_CFG, help="SAM2 model cfg, e.g. configs/sam2.1/sam2.1_hiera_l.yaml.")
    parser.add_argument(
        "--prompt-mode",
        default="mask_box_points",
        choices=["mask", "box", "points", "box_points", "mask_box_points"],
        help="Prompt construction mode for first-frame masks.",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Limit processed videos; 0 means all.")
    parser.add_argument("--resize-long-side", type=int, default=0, help="Resize inference frames by long side; 0 keeps original size.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with complete existing masks.")
    parser.add_argument("--save-raw-logits", action="store_true", help="Save raw per-object SAM2 logits as npz files.")
    parser.add_argument(
        "--save-overlays",
        default="sample",
        choices=["none", "sample", "all"],
        help="Write debug overlay JPGs. 'sample' writes every --overlay-stride frames.",
    )
    parser.add_argument("--overlay-stride", type=int, default=12, help="Frame stride used when --save-overlays=sample.")
    parser.add_argument("--make-submission", action="store_true", help="Create and validate submission.zip after inference.")
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _video_lookup(data_info: DataInfo) -> dict[str, VideoInfo]:
    """Return videos keyed by id."""

    return {video.video_id: video for video in data_info.videos}


def _expected_masks_for_video(format_spec: FormatSpec, video_id: str) -> list[dict[str, Any]]:
    """Return expected submission masks for one video."""

    return [mask for mask in format_spec.expected_masks if str(mask.get("video_id")) == video_id]


def _frame_stems_for_video(format_spec: FormatSpec, video_id: str, video_info: VideoInfo) -> list[str]:
    """Return output frame stems, preferring sample submission stems."""

    masks = sorted(_expected_masks_for_video(format_spec, video_id), key=lambda item: int(item.get("frame_index", 0)))
    if masks:
        return [str(mask["frame_stem"]) for mask in masks]
    return [frame.frame_stem for frame in video_info.frames] or [f"{idx:05d}" for idx in range(video_info.frame_count)]


def _complete_existing_video(exp_dir: Path, format_spec: FormatSpec, video: VideoInfo) -> bool:
    """Return whether all expected masks for a video already exist with correct sizes."""

    video_id = video.video_id
    masks = _expected_masks_for_video(format_spec, video_id)
    if not masks:
        stems = _frame_stems_for_video(format_spec, video_id, video)
        masks = [
            {
                "frame_stem": stem,
                "width": video.width or 0,
                "height": video.height or 0,
            }
            for stem in stems
        ]
    if not masks:
        return False
    for mask in masks:
        mask_path = exp_dir / "masks" / video_id / f"{mask['frame_stem']}.png"
        if not mask_path.exists():
            return False
        expected_width = int(mask.get("width") or video.width or 0)
        expected_height = int(mask.get("height") or video.height or 0)
        if expected_width and expected_height:
            with Image.open(mask_path) as image:
                if image.size != (expected_width, expected_height):
                    return False
    return True


def _load_status(status_path: Path) -> dict[str, Any]:
    """Load per-video status JSON if it exists."""

    if not status_path.exists():
        return {"videos": {}, "summary": {}}
    return json.loads(status_path.read_text(encoding="utf-8"))


def _select_videos(data_info: DataInfo, format_spec: FormatSpec, max_videos: int) -> tuple[list[VideoInfo], list[str]]:
    """Select videos in sample/provisional submission order."""

    videos_by_id = _video_lookup(data_info)
    missing: list[str] = []
    ordered_ids = list(format_spec.expected_videos) or [video.video_id for video in data_info.videos]
    selected: list[VideoInfo] = []
    for video_id in ordered_ids:
        video = videos_by_id.get(video_id)
        if video is None:
            missing.append(video_id)
            continue
        selected.append(video)
        if max_videos and len(selected) >= max_videos:
            break
    return selected, missing


def _subset_format_spec(format_spec: FormatSpec, video_ids: list[str], reason: str) -> FormatSpec:
    """Return a copy of a format spec restricted to selected videos."""

    keep = set(video_ids)
    expected_masks = [mask for mask in format_spec.expected_masks if str(mask.get("video_id")) in keep]
    counts = {
        video_id: count
        for video_id, count in format_spec.expected_frame_count_per_video.items()
        if video_id in keep
    }
    sizes = {
        video_id: size
        for video_id, size in format_spec.image_size_per_video.items()
        if video_id in keep
    }
    return FormatSpec(
        relative_mask_path_pattern=format_spec.relative_mask_path_pattern,
        mask_extension=format_spec.mask_extension,
        mask_encoding=format_spec.mask_encoding,
        object_id_encoding_rule=format_spec.object_id_encoding_rule,
        expected_videos=[video_id for video_id in format_spec.expected_videos if video_id in keep],
        expected_frame_count_per_video=counts,
        image_size_per_video=sizes,
        not_verified_by_sample=format_spec.not_verified_by_sample,
        expected_masks=expected_masks,
        sample_relative_paths=[path for path in format_spec.sample_relative_paths if any(f"/{video_id}/" in f"/{path}" for video_id in keep)],
        format_source=format_spec.format_source,
        notes=[*format_spec.notes, reason],
    )


def _prepare_format_spec(data_info: DataInfo, sample_submission: str | None, format_spec_path: Path) -> FormatSpec:
    """Inspect sample submission or infer a provisional format."""

    if sample_submission and Path(sample_submission).exists():
        format_spec = inspect_sample_submission(sample_submission)
    else:
        format_spec = infer_provisional_format(data_info.to_dict())
    save_format_spec(format_spec, format_spec_path)
    return format_spec


def _write_failure_sanity(exp_dir: Path, errors: list[str], warnings: list[str]) -> None:
    """Write a sanity_check.json for failed runs."""

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
    """CLI entrypoint for SAM2 baseline inference."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve()
    outputs_root = Path(args.output_dir).expanduser().resolve()
    exp_dir = outputs_root / args.experiment_id
    logs_dir = exp_dir / "logs"
    cache_dir = exp_dir / "cache"
    status_path = logs_dir / "per_video_status.json"
    data_info_path = exp_dir / "data_info.json"
    format_spec_path = exp_dir / "format_spec.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_info = inspect_dataset(data_root)
    save_data_info(data_info, data_info_path)
    format_spec = _prepare_format_spec(data_info, args.sample_submission, format_spec_path)
    selected_videos, missing_videos = _select_videos(data_info, format_spec, args.max_videos)
    if args.max_videos and selected_videos:
        format_spec = _subset_format_spec(
            format_spec,
            [video.video_id for video in selected_videos],
            reason=f"restricted_by_max_videos={args.max_videos}",
        )
        save_format_spec(format_spec, format_spec_path)
    status = _load_status(status_path)
    status.setdefault("videos", {})

    errors: list[str] = []
    warnings: list[str] = []
    if missing_videos:
        errors.append(f"Sample/provisional format references videos not found in data_info: {missing_videos}")
        for video_id in missing_videos:
            status["videos"][video_id] = {"status": "failed", "error": "video not found in data_info"}

    try:
        repo_dir = Path(args.sam2_repo_dir).expanduser().resolve() if args.sam2_repo_dir else exp_dir / "external" / "sam2"
        install_or_check_sam2(repo_dir)
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_path = Path(args.checkpoint).expanduser()
        if checkpoint_path.exists():
            checkpoint = checkpoint_path.resolve()
        else:
            checkpoint = download_sam2_checkpoint(args.checkpoint, checkpoint_dir)
        predictor = build_sam2_video_predictor(
            checkpoint_path=checkpoint,
            model_cfg=args.model_cfg,
            device="cuda",
            vos_optimized=False,
        )
    except Exception as exc:
        errors.append(f"SAM2 setup failed: {type(exc).__name__}: {exc}")
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    for video in selected_videos:
        video_id = video.video_id
        if args.skip_existing and _complete_existing_video(exp_dir, format_spec, video):
            status["videos"][video_id] = {
                "status": "skipped_existing",
                "frame_count": video.frame_count,
                "reason": "--skip-existing and masks are complete",
            }
            _atomic_write_json(status_path, status)
            continue

        run_config = {
            "data_root": str(data_root),
            "cache_dir": str(cache_dir),
            "predictor": predictor,
            "device": "cuda",
            "prompt_mode": args.prompt_mode,
            "resize_long_side": args.resize_long_side,
            "save_raw_logits": args.save_raw_logits,
            "save_overlays": args.save_overlays,
            "overlay_stride": args.overlay_stride,
            "output_frame_stems": _frame_stems_for_video(format_spec, video_id, video),
            "box_padding_ratio": 0.08,
            "num_positive": 5,
            "num_negative": 8,
            "negative_r1": 8,
            "negative_r2": 25,
        }
        result = run_sam2_on_video(video, video.prompts, exp_dir, run_config)
        status["videos"][video_id] = result.to_dict()
        if result.status != "done":
            errors.append(f"{video_id}: {result.error}")
        warnings.extend(f"{video_id}: {warning}" for warning in result.warnings)
        _atomic_write_json(status_path, status)

    if errors:
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    if args.make_submission:
        try:
            submission_path = exp_dir / "submission.zip"
            make_submission(exp_dir / "masks", submission_path, format_spec)
            sanity = validate_submission_zip(submission_path, format_spec, data_info.to_dict())
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
        "num_videos": len(selected_videos),
        "warnings": warnings,
        "submission_zip": str(exp_dir / "submission.zip") if args.make_submission else None,
    }
    _atomic_write_json(status_path, status)
    print(json.dumps(status["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

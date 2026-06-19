"""Run Cutie as an independent mask-conditioned VOS candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
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
from src.vos.cutie_optional import install_or_check_cutie, run_cutie_on_video


SMOKE_VIDEO_IDS = ["0u8fy7u2", "2b827e3a", "2a1jkxdf", "kpg9gld7", "lkob5diu", "pjlde9hu"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Cutie candidate masks for SUFE VOS.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-submission")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default=f"cutie_candidate_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--cutie-repo-dir", default=None)
    parser.add_argument("--install-cutie", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-internal-size", type=int, default=720, choices=[-1, 480, 720])
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--video-ids", default="", help="Comma-separated video IDs. Overrides --max-videos ordering when set.")
    parser.add_argument("--smoke", action="store_true", help="Run the fixed six-video smoke subset.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-overlays", default="sample", choices=["none", "sample", "all"])
    parser.add_argument("--overlay-stride", type=int, default=12)
    parser.add_argument("--make-submission", action="store_true")
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def _video_lookup(data_info: DataInfo) -> dict[str, VideoInfo]:
    return {video.video_id: video for video in data_info.videos}


def _expected_masks_for_video(format_spec: FormatSpec, video_id: str) -> list[dict[str, Any]]:
    return [mask for mask in format_spec.expected_masks if str(mask.get("video_id")) == str(video_id)]


def _frame_stems_for_video(format_spec: FormatSpec, video_id: str, video_info: VideoInfo) -> list[str]:
    masks = sorted(_expected_masks_for_video(format_spec, video_id), key=lambda item: int(item.get("frame_index", 0)))
    if masks:
        return [str(mask["frame_stem"]) for mask in masks]
    return [frame.frame_stem for frame in video_info.frames] or [f"{idx:05d}" for idx in range(video_info.frame_count)]


def _prepare_format_spec(data_info: DataInfo, sample_submission: str | None, output_path: Path) -> FormatSpec:
    if sample_submission and Path(sample_submission).exists():
        spec = inspect_sample_submission(sample_submission)
    else:
        spec = infer_provisional_format(data_info.to_dict())
    save_format_spec(spec, output_path)
    return spec


def _subset_format_spec(format_spec: FormatSpec, video_ids: list[str], reason: str) -> FormatSpec:
    keep = set(video_ids)
    return FormatSpec(
        relative_mask_path_pattern=format_spec.relative_mask_path_pattern,
        mask_extension=format_spec.mask_extension,
        mask_encoding=format_spec.mask_encoding,
        object_id_encoding_rule=format_spec.object_id_encoding_rule,
        expected_videos=[video_id for video_id in format_spec.expected_videos if video_id in keep],
        expected_frame_count_per_video={
            video_id: count for video_id, count in format_spec.expected_frame_count_per_video.items() if video_id in keep
        },
        image_size_per_video={video_id: size for video_id, size in format_spec.image_size_per_video.items() if video_id in keep},
        not_verified_by_sample=format_spec.not_verified_by_sample,
        expected_masks=[mask for mask in format_spec.expected_masks if str(mask.get("video_id")) in keep],
        sample_relative_paths=[
            path for path in format_spec.sample_relative_paths if any(f"/{video_id}/" in f"/{path}" for video_id in keep)
        ],
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


def _complete_existing_video(exp_dir: Path, format_spec: FormatSpec, video: VideoInfo) -> bool:
    masks = _expected_masks_for_video(format_spec, video.video_id)
    stems = [str(mask.get("frame_stem")) for mask in masks] if masks else _frame_stems_for_video(format_spec, video.video_id, video)
    if not stems:
        return False
    for stem in stems:
        path = exp_dir / "masks" / video.video_id / f"{stem}.png"
        if not path.exists():
            return False
        expected_width = int(video.width or 0)
        expected_height = int(video.height or 0)
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


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    exp_dir = output_root / args.experiment_id
    logs_dir = exp_dir / "logs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_info = inspect_dataset(data_root)
    save_data_info(data_info, exp_dir / "data_info.json")
    format_spec = _prepare_format_spec(data_info, args.sample_submission, exp_dir / "format_spec.json")
    selected, missing = _selected_videos(data_info, format_spec, args)
    if selected and (args.smoke or args.video_ids.strip() or args.max_videos):
        reason = "restricted_by_cutie_smoke_or_selection"
        format_spec = _subset_format_spec(format_spec, [video.video_id for video in selected], reason)
        save_format_spec(format_spec, exp_dir / "format_spec.json")

    status: dict[str, Any] = {"videos": {}, "summary": {}}
    errors: list[str] = []
    warnings: list[str] = []
    if args.make_submission and (not args.sample_submission or not Path(args.sample_submission).expanduser().exists()):
        errors.append("--make-submission requires an existing --sample-submission for strict format validation")
    if not data_info.videos:
        errors.append(
            f"No videos were found under --data-root {data_root}; "
            f"scan saw {int(data_info.scan.get('num_files', 0))} files"
        )
    if not selected:
        errors.append("No videos selected for Cutie inference")
    if missing:
        errors.append(f"Requested videos not found: {missing}")
    if errors:
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_json(logs_dir / "per_video_status.json", status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    cutie_status = install_or_check_cutie(args.cutie_repo_dir, install=bool(args.install_cutie))
    if not cutie_status.available:
        errors.append(f"Cutie setup failed: {cutie_status.reason}; {cutie_status.error or ''}".strip())
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings, "cutie": cutie_status.to_dict()}
        _atomic_json(logs_dir / "per_video_status.json", status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    model: Any | None = None
    if not errors:
        try:
            from src.vos.cutie_optional import build_cutie_model

            model = build_cutie_model(device=args.device)
        except Exception as exc:
            errors.append(f"Cutie model build failed: {type(exc).__name__}: {exc}")

    for video in selected:
        if errors:
            break
        if args.skip_existing and _complete_existing_video(exp_dir, format_spec, video):
            status["videos"][video.video_id] = {
                "status": "skipped_existing",
                "frame_count": video.frame_count,
                "reason": "--skip-existing and masks are complete",
            }
            _atomic_json(logs_dir / "per_video_status.json", status)
            continue
        config = {
            "data_root": str(data_root),
            "cache_dir": str(exp_dir / "cache"),
            "device": args.device,
            "cutie_model": model,
            "max_internal_size": args.max_internal_size,
            "resize_long_side": 0,
            "max_frames": args.max_frames,
            "output_frame_stems": _frame_stems_for_video(format_spec, video.video_id, video),
            "save_overlays": args.save_overlays,
            "overlay_stride": args.overlay_stride,
        }
        result = run_cutie_on_video(video, video.prompts, exp_dir, config)
        status["videos"][video.video_id] = result.to_dict()
        if result.status != "done":
            errors.append(f"{video.video_id}: {result.error}")
        warnings.extend(f"{video.video_id}: {warning}" for warning in result.warnings)
        _atomic_json(logs_dir / "per_video_status.json", status)

    if errors:
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings, "cutie": cutie_status.to_dict()}
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
                status["summary"] = {"status": "failed_validation", "errors": sanity["errors"], "warnings": sanity["warnings"]}
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
                "num_videos": len(selected),
                "candidate_only": True,
            },
        )

    manifest = {
        "experiment_id": args.experiment_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": "cutie_optional",
        "candidate_only": not bool(args.make_submission),
        "max_internal_size": args.max_internal_size,
        "first_frame_policy": "copy_input_annotation_exactly",
        "object_id_policy": "preserve_indexed_prompt_ids",
        "selected_videos": [video.video_id for video in selected],
        "cutie": cutie_status.to_dict(),
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

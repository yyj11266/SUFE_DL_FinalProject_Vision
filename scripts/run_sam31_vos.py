"""Run a strict native SAM 3.1 Object Multiplex VOS baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
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
from src.trackers.sam3_tracker_optional import (
    SAM31_HF_REPO,
    SAM3_EMPTY_MASK_POLICIES,
    SAM3_EMPTY_MASK_POLICY_EMPTY,
    SAM3_RUN_MODE_FULL,
    SAM3_RUN_MODE_LOW_LEVEL,
    SAM3_RUN_MODES,
    build_sam3_tracker,
    install_sam3_if_requested,
    run_sam3_video_with_mask_prompt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native SAM 3.1 multiplex VOS with full first-frame masks.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sample-submission")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default=f"sam31_native_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--checkpoint", help=f"Local SAM 3.1 checkpoint. If omitted, download from --hf-repo-id.")
    parser.add_argument("--hf-repo-id", default=SAM31_HF_REPO, help="Hugging Face repo containing sam3.1_multiplex.pt.")
    parser.add_argument("--checkpoint-filename", default="sam3.1_multiplex.pt")
    parser.add_argument("--sam3-repo-dir", help="Existing official SAM3 clone or installation target.")
    parser.add_argument("--install-sam3", action="store_true", help="Clone/install official SAM3 if needed.")
    parser.add_argument("--prompt-mode", default="mask", choices=["mask"])
    parser.add_argument(
        "--sam3-run-mode",
        default=SAM3_RUN_MODE_FULL,
        choices=SAM3_RUN_MODES,
        help="SAM 3.1 backend mode. Only full_predictor_mask is allowed for submissions.",
    )
    parser.add_argument("--original-resolution", action="store_true", help="Explicitly document original-resolution inference.")
    parser.add_argument("--video-ids", help="Comma-separated video IDs for smoke or split runs.")
    parser.add_argument("--video-ids-file", help="JSON list, split JSON, or newline-separated video IDs.")
    parser.add_argument("--split", choices=["calibration", "holdout"], help="Key to read from --video-ids-file split JSON.")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Smoke tests only; 0 runs all frames.")
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--use-fa3", action="store_true", help="Enable FlashAttention 3 on a compatible H100 runtime.")
    parser.add_argument("--use-rope-real", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument(
        "--disable-internal-tracker-recovery",
        action="store_true",
        help="Development only; do not recover missing full-predictor objects from internal SAM2 tracker logits.",
    )
    parser.add_argument(
        "--sam3-empty-mask-policy",
        default=SAM3_EMPTY_MASK_POLICY_EMPTY,
        choices=SAM3_EMPTY_MASK_POLICIES,
        help="Development diagnostic. 'previous' holds the previous object mask when SAM3 emits an empty mask.",
    )
    parser.add_argument("--save-native-scores", action="store_true")
    parser.add_argument("--save-raw-logits", action="store_true")
    parser.add_argument(
        "--save-overlays",
        default="sample",
        choices=["none", "sample", "all"],
        help="Write debug overlay JPGs. 'sample' writes every --overlay-stride frames.",
    )
    parser.add_argument("--overlay-stride", type=int, default=12, help="Frame stride used when --save-overlays=sample.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-unsupported-runtime", action="store_true", help="Development only; bypass version checks.")
    parser.add_argument("--smoke-quality-gate", action="store_true", help="Fail short SAM3.1 smoke runs when masks collapse or explode.")
    parser.add_argument("--quality-gate", action="store_true", help="Run object-level full-run quality checks before creating a submission.")
    parser.add_argument("--disable-quality-gate", action="store_true", help="Development only; skip the submission quality gate.")
    parser.add_argument("--quality-gate-baseline-exp", help="Baseline experiment directory used for empty-mask comparison.")
    parser.add_argument("--quality-gate-max-extra-empty-frames", type=int, default=100)
    parser.add_argument("--quality-gate-max-extra-empty-ratio", type=float, default=0.05)
    parser.add_argument("--quality-gate-severe-zero-ratio", type=float, default=0.95)
    parser.add_argument("--quality-gate-early-frame-window", type=int, default=20)
    parser.add_argument("--make-submission", action="store_true")
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _format_spec(data_info: DataInfo, sample_submission: str | None) -> FormatSpec:
    if sample_submission and Path(sample_submission).expanduser().exists():
        return inspect_sample_submission(sample_submission)
    return infer_provisional_format(data_info.to_dict())


def _load_video_ids(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.video_ids:
        requested.extend(value.strip() for value in args.video_ids.split(",") if value.strip())
    if args.video_ids_file:
        path = Path(args.video_ids_file).expanduser()
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(payload, dict):
            key = args.split or "video_ids"
            payload = payload.get(key, [])
        if not isinstance(payload, list):
            raise ValueError(f"Video ID file must contain a list or split mapping: {path}")
        requested.extend(str(value) for value in payload)
    return list(dict.fromkeys(requested))


def _select_videos(data_info: DataInfo, spec: FormatSpec, requested: list[str], max_videos: int) -> list[VideoInfo]:
    by_id = {video.video_id: video for video in data_info.videos}
    ordered = requested or list(spec.expected_videos) or [video.video_id for video in data_info.videos]
    missing = [video_id for video_id in ordered if video_id not in by_id]
    if missing:
        raise RuntimeError(f"Requested videos are absent from the dataset: {missing}")
    selected = [by_id[video_id] for video_id in ordered]
    return selected[:max_videos] if max_videos else selected


def _subset_spec(spec: FormatSpec, video_ids: list[str]) -> FormatSpec:
    keep = set(video_ids)
    return replace(
        spec,
        expected_videos=[video_id for video_id in spec.expected_videos if video_id in keep],
        expected_frame_count_per_video={key: value for key, value in spec.expected_frame_count_per_video.items() if key in keep},
        image_size_per_video={key: value for key, value in spec.image_size_per_video.items() if key in keep},
        expected_masks=[mask for mask in spec.expected_masks if str(mask.get("video_id")) in keep],
        sample_relative_paths=[
            path for path in spec.sample_relative_paths if any(f"/{video_id}/" in f"/{path}" for video_id in keep)
        ],
        notes=[*spec.notes, "restricted_video_subset"],
    )


def _expected_stems(spec: FormatSpec, video: VideoInfo) -> list[str]:
    entries = [mask for mask in spec.expected_masks if str(mask.get("video_id")) == video.video_id]
    entries.sort(key=lambda item: int(item.get("frame_index", 0)))
    return [str(item["frame_stem"]) for item in entries] or [frame.frame_stem for frame in video.frames]


def _existing_complete(exp_dir: Path, spec: FormatSpec, video: VideoInfo, data_root: Path) -> bool:
    stems = _expected_stems(spec, video)
    if not stems:
        return False
    prompt = next((item for item in video.prompts if item.prompt_type == "mask"), None)
    if prompt is None:
        return False
    expected_first = data_root / prompt.relative_path
    if not expected_first.exists():
        return False
    with Image.open(expected_first) as image:
        expected_first_array = image.convert("L").copy()
    known_ids = set(prompt.object_ids)
    for stem in stems:
        path = exp_dir / "masks" / video.video_id / f"{stem}.png"
        if not path.exists():
            return False
        with Image.open(path) as image:
            if video.width and video.height and image.size != (video.width, video.height):
                return False
            values = set(int(value) for value in image.convert("L").getdata())
        if known_ids and not (values - {0}).issubset(known_ids):
            return False
    with Image.open(exp_dir / "masks" / video.video_id / f"{stems[0]}.png") as image:
        if list(image.convert("L").getdata()) != list(expected_first_array.getdata()):
            return False
    return True


def _foreground_stats(mask_path: Path, object_ids: set[int]) -> dict[str, Any]:
    with Image.open(mask_path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    values = set(int(value) for value in np.unique(array))
    foreground = array > 0
    return {
        "path": str(mask_path),
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "foreground_pixels": int(foreground.sum()),
        "foreground_fraction": float(foreground.mean()),
        "object_ids_present": sorted(values - {0}),
        "unexpected_object_ids": sorted((values - {0}) - object_ids),
    }


def _resolve_recorded_mask_path(exp_dir: Path, video_id: str, mask_path_text: str) -> Path:
    path = Path(mask_path_text)
    if path.exists():
        return path
    return exp_dir / "masks" / video_id / path.name


def _empty_mask_summary(exp_dir: Path, status: dict[str, Any] | None = None) -> dict[str, Any]:
    videos: dict[str, Any] = {}
    total_non_first_empty_frames = 0
    total_frames = 0
    if status is not None:
        video_items = status.get("videos", {}).items()
    else:
        video_items = ((path.name, {"mask_paths": [str(item) for item in sorted(path.glob("*.png"))]}) for path in sorted((exp_dir / "masks").glob("*")))
    for video_id, result in video_items:
        if status is not None and result.get("status") not in {"done", "skipped_existing"}:
            continue
        paths = [
            _resolve_recorded_mask_path(exp_dir, str(video_id), str(path))
            for path in result.get("mask_paths", [])
        ]
        if not paths:
            paths = sorted((exp_dir / "masks" / str(video_id)).glob("*.png"))
        empty_non_first = 0
        max_foreground_fraction = 0.0
        min_foreground_fraction: float | None = None
        for index, path in enumerate(paths):
            with Image.open(path) as image:
                foreground = np.asarray(image.convert("L"), dtype=np.uint8) > 0
            fraction = float(foreground.mean())
            max_foreground_fraction = max(max_foreground_fraction, fraction)
            min_foreground_fraction = fraction if min_foreground_fraction is None else min(min_foreground_fraction, fraction)
            if index > 0 and int(foreground.sum()) == 0:
                empty_non_first += 1
        total_non_first_empty_frames += empty_non_first
        total_frames += len(paths)
        videos[str(video_id)] = {
            "frame_count": len(paths),
            "non_first_empty_frames": empty_non_first,
            "min_foreground_fraction": min_foreground_fraction,
            "max_foreground_fraction": max_foreground_fraction,
        }
    return {
        "total_frames": total_frames,
        "total_non_first_empty_frames": total_non_first_empty_frames,
        "videos": videos,
    }


def _object_quality_errors(
    status: dict[str, Any],
    severe_zero_ratio: float,
    early_frame_window: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    severe_objects: list[dict[str, Any]] = []
    for video_id, result in status.get("videos", {}).items():
        diagnostics = result.get("diagnostics") or {}
        per_object = diagnostics.get("per_object") or {}
        for object_id, item in per_object.items():
            total_frames = int(item.get("total_frames", 0) or 0)
            if total_frames <= 1:
                continue
            first_zero = item.get("first_zero_frame")
            zero_ratio = float(item.get("zero_ratio", 0.0) or 0.0)
            recovers_after_zero = bool(item.get("recovers_after_zero", False))
            missing_frames = int(item.get("missing_output_frames", 0) or 0)
            first_missing = item.get("first_missing_output_frame")
            early_permanent_zero = (
                first_zero is not None
                and 0 < int(first_zero) < min(total_frames - 1, early_frame_window)
                and not recovers_after_zero
            )
            severe_zero = zero_ratio >= severe_zero_ratio and first_zero is not None and int(first_zero) <= early_frame_window
            if early_permanent_zero or severe_zero or missing_frames:
                payload = {
                    "video_id": video_id,
                    "object_id": int(object_id),
                    "total_frames": total_frames,
                    "zero_ratio": zero_ratio,
                    "first_zero_frame": first_zero,
                    "recovers_after_zero": recovers_after_zero,
                    "missing_output_frames": missing_frames,
                    "first_missing_output_frame": first_missing,
                }
                severe_objects.append(payload)
                if missing_frames:
                    errors.append(
                        f"{video_id} obj{object_id}: full predictor omitted object ID in {missing_frames} frame(s); "
                        f"first missing frame={first_missing}"
                    )
                elif early_permanent_zero:
                    errors.append(
                        f"{video_id} obj{object_id}: object becomes zero at frame {first_zero} within the first "
                        f"{early_frame_window} frames and never recovers"
                    )
                else:
                    errors.append(
                        f"{video_id} obj{object_id}: zero-mask ratio {zero_ratio:.3f} exceeds {severe_zero_ratio:.3f}; "
                        f"first zero frame={first_zero}"
                    )
    return errors, severe_objects


def _write_contact_sheet(video_id: str, overlay_paths: list[str], output_path: Path, max_frames: int = 5) -> str | None:
    selected = [Path(path) for path in overlay_paths[:max_frames] if Path(path).exists()]
    if not selected:
        return None
    thumbs: list[Image.Image] = []
    for path in selected:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((320, 240))
            canvas = Image.new("RGB", (320, 240), (20, 20, 20))
            canvas.paste(thumb, ((320 - thumb.width) // 2, (240 - thumb.height) // 2))
            thumbs.append(canvas)
    sheet = Image.new("RGB", (320 * len(thumbs), 240), (20, 20, 20))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (320 * index, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _run_smoke_quality_gate(
    exp_dir: Path,
    status: dict[str, Any],
    max_foreground_fraction: float = 0.60,
    tiny_initial_foreground_fraction: float = 0.001,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    videos_payload: dict[str, Any] = {}
    contact_sheets: dict[str, str] = {}
    for video_id, result in status.get("videos", {}).items():
        if result.get("status") != "done":
            continue
        object_ids = {int(value) for value in result.get("object_ids", [])}
        frame_stats: list[dict[str, Any]] = []
        empty_non_first: list[str] = []
        for mask_path_text in result.get("mask_paths", []):
            stats = _foreground_stats(Path(mask_path_text), object_ids)
            frame_stats.append(stats)
            if stats["unexpected_object_ids"]:
                errors.append(f"{video_id}: unexpected object IDs in {stats['path']}: {stats['unexpected_object_ids']}")
            if frame_stats and len(frame_stats) > 1:
                if stats["foreground_pixels"] == 0:
                    empty_non_first.append(str(stats["path"]))
                if stats["foreground_fraction"] > max_foreground_fraction:
                    errors.append(
                        f"{video_id}: non-first smoke frame foreground fraction {stats['foreground_fraction']:.3f} "
                        f"exceeds {max_foreground_fraction:.3f}: {stats['path']}"
                    )
        if not result.get("first_frame_exact", False):
            errors.append(f"{video_id}: first frame was not pixel-identical to the prompt mask")
        initial_foreground_fraction = float(frame_stats[0]["foreground_fraction"]) if frame_stats else 0.0
        is_tiny_single_object = len(object_ids) <= 1 and initial_foreground_fraction < tiny_initial_foreground_fraction
        if empty_non_first:
            if len(object_ids) > 1:
                errors.append(f"{video_id}: multi-object smoke collapsed to empty masks: {empty_non_first[:5]}")
            elif is_tiny_single_object:
                warnings.append(
                    f"{video_id}: tiny single-object smoke has empty non-first frame(s); "
                    f"kept as warning because initial foreground fraction is {initial_foreground_fraction:.6f}: "
                    f"{empty_non_first[:5]}"
                )
            else:
                errors.append(f"{video_id}: non-first smoke frame is empty: {empty_non_first[:5]}")
        object_errors, severe_objects = _object_quality_errors(
            {"videos": {video_id: result}},
            severe_zero_ratio=0.95,
            early_frame_window=20,
        )
        if object_errors:
            errors.extend(object_errors)
        sheet = _write_contact_sheet(
            video_id,
            [str(path) for path in result.get("overlay_paths", [])],
            exp_dir / "logs" / "smoke_contact_sheets" / f"{video_id}.jpg",
        )
        if sheet:
            contact_sheets[video_id] = sheet
        videos_payload[video_id] = {
            "object_ids": sorted(object_ids),
            "frames": frame_stats,
            "contact_sheet": sheet,
            "severe_objects": severe_objects,
        }
    if not videos_payload:
        errors.append("Smoke quality gate found no completed videos to inspect")
    payload = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "max_foreground_fraction": max_foreground_fraction,
        "tiny_initial_foreground_fraction": tiny_initial_foreground_fraction,
        "contact_sheets": contact_sheets,
        "videos": videos_payload,
    }
    _atomic_json(exp_dir / "logs" / "smoke_quality_gate.json", payload)
    return payload


def _run_full_quality_gate(
    exp_dir: Path,
    status: dict[str, Any],
    baseline_exp: Path | None,
    max_extra_empty_frames: int,
    max_extra_empty_ratio: float,
    severe_zero_ratio: float,
    early_frame_window: int,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    current = _empty_mask_summary(exp_dir, status)
    baseline = None
    if baseline_exp and baseline_exp.exists() and (baseline_exp / "masks").exists():
        baseline = _empty_mask_summary(baseline_exp)
        allowed_extra = max(int(max_extra_empty_frames), int(current["total_frames"] * float(max_extra_empty_ratio)))
        extra_empty = int(current["total_non_first_empty_frames"]) - int(baseline["total_non_first_empty_frames"])
        if extra_empty > allowed_extra:
            errors.append(
                "SAM3.1 generated too many extra non-first empty masks compared with the baseline: "
                f"current={current['total_non_first_empty_frames']}, baseline={baseline['total_non_first_empty_frames']}, "
                f"extra={extra_empty}, allowed={allowed_extra}"
            )
    else:
        warnings.append(
            "Baseline mask directory was not found; full quality gate skipped baseline empty-mask comparison."
        )

    object_errors, severe_objects = _object_quality_errors(status, severe_zero_ratio, early_frame_window)
    errors.extend(object_errors)
    payload = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "current": current,
        "baseline": baseline,
        "baseline_exp": str(baseline_exp) if baseline_exp else None,
        "max_extra_empty_frames": int(max_extra_empty_frames),
        "max_extra_empty_ratio": float(max_extra_empty_ratio),
        "severe_zero_ratio": float(severe_zero_ratio),
        "early_frame_window": int(early_frame_window),
        "severe_objects": severe_objects,
    }
    _atomic_json(exp_dir / "logs" / "full_quality_gate.json", payload)
    return payload


def _resolve_checkpoint(args: argparse.Namespace, exp_dir: Path) -> Path | None:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        return path.resolve() if path.exists() else path
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download SAM 3.1 checkpoint; "
            "install requirements_colab.txt or pass --checkpoint."
        ) from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    cache_dir = exp_dir / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.checkpoint_filename,
            token=token,
            local_dir=str(cache_dir),
        )
    ).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.make_submission and args.sam3_run_mode != SAM3_RUN_MODE_FULL:
        raise SystemExit("--make-submission is only allowed with --sam3-run-mode full_predictor_mask")
    if args.make_submission and not args.sample_submission:
        raise SystemExit("--make-submission requires --sample-submission for strict format validation")
    if args.make_submission and args.sample_submission and not Path(args.sample_submission).expanduser().exists():
        raise SystemExit(f"--sample-submission does not exist: {args.sample_submission}")
    if args.make_submission and args.max_frames:
        raise SystemExit("--make-submission cannot be combined with --max-frames")
    if args.make_submission and (args.max_videos or args.video_ids or args.video_ids_file):
        raise SystemExit("Partial video selection cannot create a final submission")

    data_root = Path(args.data_root).expanduser().resolve()
    exp_dir = Path(args.output_dir).expanduser().resolve() / args.experiment_id
    logs_dir = exp_dir / "logs"
    cache_dir = exp_dir / "cache"
    status_path = logs_dir / "per_video_status.json"
    manifest_path = exp_dir / "run_manifest.json"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    data_info = inspect_dataset(data_root)
    save_data_info(data_info, exp_dir / "data_info.json")
    spec = _format_spec(data_info, args.sample_submission)
    requested = _load_video_ids(args)
    selected = _select_videos(data_info, spec, requested, args.max_videos)
    if requested or args.max_videos:
        spec = _subset_spec(spec, [video.video_id for video in selected])
    save_format_spec(spec, exp_dir / "format_spec.json")
    print(
        json.dumps(
            {
                "stage": "prepared_data",
                "num_selected_videos": len(selected),
                "selected_videos": [video.video_id for video in selected],
                "experiment_dir": str(exp_dir),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    if args.sam3_repo_dir:
        repo_dir = Path(args.sam3_repo_dir).expanduser().resolve()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
    if args.install_sam3:
        install_dir = args.sam3_repo_dir or str(exp_dir / "external" / "sam3")
        install_status = install_sam3_if_requested(install_dir, requested=True)
        if not install_status.available:
            _atomic_json(status_path, {"summary": {"status": "failed_setup", "error": install_status.reason}, "videos": {}})
            print(json.dumps(install_status.to_dict(), indent=2, ensure_ascii=True))
            return 1

    try:
        checkpoint_path = _resolve_checkpoint(args, exp_dir)
    except Exception as exc:
        status = {"summary": {"status": "failed_checkpoint_download", "error": f"{type(exc).__name__}: {exc}"}, "videos": {}}
        _atomic_json(status_path, status)
        print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
        return 1
    print(json.dumps({"stage": "checkpoint_ready", "checkpoint": str(checkpoint_path)}, ensure_ascii=True), flush=True)

    build = build_sam3_tracker(
        checkpoint_path=checkpoint_path,
        device="cuda",
        multiplex_count=args.multiplex_count,
        use_fa3=args.use_fa3,
        use_rope_real=args.use_rope_real,
        compile_model=args.compile,
        strict_runtime=not args.allow_unsupported_runtime,
        run_mode=args.sam3_run_mode,
    )
    manifest = {
        "experiment_id": args.experiment_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "backend": "sam3.1_object_multiplex",
        "sam3_run_mode": args.sam3_run_mode,
        "mask_conditioning_path": (
            "official_full_predictor_private_tracker_add_new_objects"
            if args.sam3_run_mode == SAM3_RUN_MODE_FULL
            else "low_level_debug_add_new_masks_rejected_for_submission"
        ),
        "prompt_mode": "mask",
        "original_resolution": True,
        "first_frame_policy": "copy_input_annotation_exactly",
        "fallback_policy": "recover_internal_tracker_logits_before_quality_gate"
        if not args.disable_internal_tracker_recovery
        else "fail_without_fallback",
        "internal_tracker_recovery_enabled": not args.disable_internal_tracker_recovery,
        "empty_mask_policy": args.sam3_empty_mask_policy,
        "native_scores_requested": bool(args.save_native_scores),
        "save_overlays": args.save_overlays,
        "overlay_stride": args.overlay_stride,
        "native_score_fields": {
            "presence": "full predictor out_probs converted to logits when available; low-level object_score_logits otherwise",
            "predicted_iou": "best-effort nullable field from native output when exposed",
            "object_state": "derived from presence score when available",
        },
        "selected_videos": [video.video_id for video in selected],
        "max_frames": args.max_frames,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "hf_repo_id": args.hf_repo_id,
        "checkpoint_filename": args.checkpoint_filename,
        "arguments": vars(args),
        "build": build.to_dict(),
    }
    _atomic_json(manifest_path, manifest)
    if not build.available or build.predictor is None:
        status = {"summary": {"status": "failed_setup", "error": build.error, "warnings": build.warnings}, "videos": {}}
        _atomic_json(status_path, status)
        print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
        return 1
    print(json.dumps({"stage": "model_built", "warnings": build.warnings}, ensure_ascii=True), flush=True)

    status: dict[str, Any] = {"videos": {}, "summary": {"status": "running"}}
    _atomic_json(status_path, status)
    errors: list[str] = []
    warnings: list[str] = []
    for video in selected:
        if args.skip_existing and not args.max_frames and _existing_complete(exp_dir, spec, video, data_root):
            status["videos"][video.video_id] = {
                "status": "skipped_existing",
                "frame_count": video.frame_count,
                "first_frame_exact": True,
            }
            _atomic_json(status_path, status)
            continue
        result = run_sam3_video_with_mask_prompt(
            video,
            video.prompts,
            exp_dir,
            {
                "data_root": str(data_root),
                "cache_dir": str(cache_dir),
                "predictor": build.predictor,
                "device": "cuda",
                "sam3_run_mode": args.sam3_run_mode,
                "prompt_mode": "mask",
                "resize_long_side": 0,
                "output_frame_stems": _expected_stems(spec, video),
                "max_frames": args.max_frames,
                "offload_video_to_cpu": args.offload_video_to_cpu,
                "offload_state_to_cpu": args.offload_state_to_cpu,
                "sam3_recover_internal_tracker_outputs": not args.disable_internal_tracker_recovery,
                "sam3_empty_mask_policy": args.sam3_empty_mask_policy,
                "save_native_scores": args.save_native_scores,
                "save_raw_logits": args.save_raw_logits,
                "save_overlays": args.save_overlays,
                "overlay_stride": args.overlay_stride,
            },
        )
        status["videos"][video.video_id] = result.to_dict()
        print(json.dumps({"stage": "video_done", "video_id": video.video_id, "status": result.status, "error": result.error}, ensure_ascii=True), flush=True)
        if result.status != "done":
            errors.append(f"{video.video_id}: {result.error}")
        warnings.extend(f"{video.video_id}: {warning}" for warning in result.warnings)
        _atomic_json(status_path, status)

    if errors:
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_json(status_path, status)
        _atomic_json(exp_dir / "sanity_check.json", {"passed": False, "errors": errors, "warnings": warnings})
        print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
        return 1

    smoke_gate: dict[str, Any] | None = None
    if args.smoke_quality_gate:
        smoke_gate = _run_smoke_quality_gate(exp_dir, status)
        if not smoke_gate["passed"]:
            status["summary"] = {"status": "failed_smoke_quality_gate", **smoke_gate}
            _atomic_json(status_path, status)
            _atomic_json(exp_dir / "sanity_check.json", {"passed": False, **smoke_gate})
            print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
            return 1
        warnings.extend(f"smoke_quality_gate: {warning}" for warning in smoke_gate.get("warnings", []))

    quality_gate: dict[str, Any] | None = None
    should_run_quality_gate = bool(args.quality_gate or (args.make_submission and not args.disable_quality_gate))
    if should_run_quality_gate:
        baseline_exp = (
            Path(args.quality_gate_baseline_exp).expanduser().resolve()
            if args.quality_gate_baseline_exp
            else Path(args.output_dir).expanduser().resolve() / "data_prep_20260608_065509"
        )
        quality_gate = _run_full_quality_gate(
            exp_dir=exp_dir,
            status=status,
            baseline_exp=baseline_exp,
            max_extra_empty_frames=args.quality_gate_max_extra_empty_frames,
            max_extra_empty_ratio=args.quality_gate_max_extra_empty_ratio,
            severe_zero_ratio=args.quality_gate_severe_zero_ratio,
            early_frame_window=args.quality_gate_early_frame_window,
        )
        if not quality_gate["passed"]:
            status["summary"] = {"status": "failed_quality_gate", **quality_gate}
            _atomic_json(status_path, status)
            _atomic_json(exp_dir / "sanity_check.json", {"passed": False, **quality_gate})
            print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
            return 1
        warnings.extend(f"full_quality_gate: {warning}" for warning in quality_gate.get("warnings", []))

    sanity: dict[str, Any] | None = None
    if args.make_submission:
        submission_path = exp_dir / "submission.zip"
        make_submission(exp_dir / "masks", submission_path, spec)
        sanity = validate_submission_zip(submission_path, spec, data_info.to_dict())
        _atomic_json(exp_dir / "sanity_check.json", sanity)
        if not sanity["passed"]:
            status["summary"] = {"status": "failed_validation", **sanity}
            _atomic_json(status_path, status)
            return 1

    status["summary"] = {
        "status": "done",
        "num_videos": len(selected),
        "num_frames": sum(int(item.get("frame_count", 0)) for item in status["videos"].values()),
        "all_first_frames_exact": all(
            bool(item.get("first_frame_exact", False))
            for item in status["videos"].values()
            if item.get("status") in {"done", "skipped_existing"}
        ),
        "submission_zip": str(exp_dir / "submission.zip") if args.make_submission else None,
        "sanity_passed": sanity["passed"] if sanity else None,
        "smoke_quality_gate_passed": smoke_gate["passed"] if smoke_gate else None,
        "full_quality_gate_passed": quality_gate["passed"] if quality_gate else None,
        "warnings": warnings,
    }
    manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["result"] = status["summary"]
    _atomic_json(manifest_path, manifest)
    _atomic_json(status_path, status)
    print(json.dumps(status["summary"], indent=2, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

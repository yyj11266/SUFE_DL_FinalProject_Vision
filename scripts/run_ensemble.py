"""Run optional model ensemble over precomputed SUFE VOS predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image


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
from src.vos.ensemble import (
    PredictionSetConfig,
    collect_object_ids_for_video,
    normalize_prediction_sets,
    parse_prediction_set_config,
    run_ensemble_for_video,
    write_ensemble_debug_csv,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for optional model ensemble."""

    parser = argparse.ArgumentParser(description="Fuse optional model prediction roots into a SUFE submission.")
    parser.add_argument("--data-root", default=None, help="Extracted SUFE data root; required if --data-info is absent.")
    parser.add_argument("--sample-submission", default=None, help="Optional sample_submission.zip path.")
    parser.add_argument("--format-spec", default=None, help="Optional precomputed format_spec.json path.")
    parser.add_argument("--data-info", default=None, help="Optional precomputed data_info.json path.")
    parser.add_argument("--output-dir", required=True, help="Outputs root directory.")
    parser.add_argument("--experiment-id", default=f"ensemble_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument(
        "--prediction-root",
        action="append",
        required=True,
        help="Prediction root, repeatable. Use name=/path/to/exp or just /path/to/exp.",
    )
    parser.add_argument("--max-videos", type=int, default=0, help="Limit processed videos; 0 means all.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos with complete existing ensemble masks.")
    parser.add_argument("--make-submission", action="store_true", help="Create and validate submission.zip after ensemble.")
    return parser


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON dictionary from disk."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_format_spec(path: str | Path) -> FormatSpec:
    """Load a ``FormatSpec`` from JSON."""

    return FormatSpec.from_dict(_load_json(path))


def _prepare_data_info(args: argparse.Namespace, exp_dir: Path) -> dict[str, Any]:
    """Load or inspect dataset info and save it under the experiment directory."""

    output_path = exp_dir / "data_info.json"
    if args.data_info:
        info = _load_json(args.data_info)
        _atomic_write_json(output_path, info)
        return info
    if not args.data_root:
        raise ValueError("--data-root is required when --data-info is not provided.")
    data_info = inspect_dataset(Path(args.data_root).expanduser().resolve())
    save_data_info(data_info, output_path)
    return data_info.to_dict()


def _prepare_format_spec(args: argparse.Namespace, data_info: dict[str, Any], exp_dir: Path) -> FormatSpec:
    """Load, inspect, or infer the submission format contract."""

    output_path = exp_dir / "format_spec.json"
    if args.format_spec:
        spec = _load_format_spec(args.format_spec)
    elif args.sample_submission and Path(args.sample_submission).exists():
        spec = inspect_sample_submission(args.sample_submission)
    else:
        spec = infer_provisional_format(data_info)
    save_format_spec(spec, output_path)
    return spec


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


def _expected_masks_by_video(format_spec: FormatSpec) -> dict[str, list[dict[str, Any]]]:
    """Group expected masks by video id."""

    grouped: dict[str, list[dict[str, Any]]] = {video_id: [] for video_id in format_spec.expected_videos}
    for mask in format_spec.expected_masks:
        grouped.setdefault(str(mask.get("video_id")), []).append(dict(mask))
    return grouped


def _select_videos(format_spec: FormatSpec, max_videos: int) -> list[str]:
    """Select video ids in format-spec order."""

    videos = list(format_spec.expected_videos) or sorted({str(mask.get("video_id")) for mask in format_spec.expected_masks})
    return videos[:max_videos] if max_videos else videos


def _complete_existing_video(exp_dir: Path, masks: list[dict[str, Any]], video_id: str) -> bool:
    """Return whether all expected ensemble masks already exist with expected sizes."""

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


def _parse_prediction_roots(values: list[str]) -> list[PredictionSetConfig]:
    """Parse repeatable prediction-root CLI values."""

    return normalize_prediction_sets(parse_prediction_set_config(value) for value in values)


def _write_failure_sanity(exp_dir: Path, errors: list[str], warnings: list[str]) -> None:
    """Write a failed sanity_check.json when submission creation cannot run."""

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
    """Run optional model ensemble CLI."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    outputs_root = Path(args.output_dir).expanduser().resolve()
    exp_dir = outputs_root / args.experiment_id
    logs_dir = exp_dir / "logs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = logs_dir / "per_video_status.json"

    errors: list[str] = []
    warnings: list[str] = []
    status: dict[str, Any] = {"videos": {}, "summary": {}}

    try:
        data_info = _prepare_data_info(args, exp_dir)
        format_spec = _prepare_format_spec(args, data_info, exp_dir)
        selected_videos = _select_videos(format_spec, args.max_videos)
        if args.max_videos:
            format_spec = _subset_format_spec(format_spec, selected_videos, f"restricted_by_max_videos={args.max_videos}")
            save_format_spec(format_spec, exp_dir / "format_spec.json")
        expected_by_video = _expected_masks_by_video(format_spec)
        prediction_sets = _parse_prediction_roots(args.prediction_root)
    except Exception as exc:
        errors.append(f"Setup failed: {type(exc).__name__}: {exc}")
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    disabled = [cfg.name for cfg in prediction_sets if not cfg.enabled]
    if disabled:
        warnings.append(f"Disabled missing prediction roots: {disabled}")
    enabled_sets = [cfg for cfg in prediction_sets if cfg.enabled]
    if not enabled_sets:
        errors.append("No enabled prediction roots were found.")
        status["summary"] = {"status": "failed", "errors": errors, "warnings": warnings}
        _atomic_write_json(status_path, status)
        _write_failure_sanity(exp_dir, errors, warnings)
        return 1

    all_debug_rows: list[dict[str, Any]] = []
    for video_id in selected_videos:
        expected_masks = expected_by_video.get(video_id, [])
        if not expected_masks:
            errors.append(f"{video_id}: no expected masks in format spec")
            status["videos"][video_id] = {"status": "failed", "error": "no expected masks"}
            _atomic_write_json(status_path, status)
            continue
        if args.skip_existing and _complete_existing_video(exp_dir, expected_masks, video_id):
            status["videos"][video_id] = {
                "status": "skipped_existing",
                "frame_count": len(expected_masks),
                "reason": "--skip-existing and masks are complete",
            }
            _atomic_write_json(status_path, status)
            continue
        object_ids = collect_object_ids_for_video(video_id, enabled_sets, data_info)
        result = run_ensemble_for_video(
            video_id=video_id,
            expected_masks=expected_masks,
            object_ids=object_ids,
            prediction_sets=enabled_sets,
            output_dir=exp_dir,
            format_spec=format_spec,
            data_info=data_info,
            skip_existing=args.skip_existing,
        )
        status["videos"][video_id] = result.to_dict()
        all_debug_rows.extend(result.debug_rows)
        warnings.extend(f"{video_id}: {warning}" for warning in result.warnings)
        if result.status != "done":
            errors.append(f"{video_id}: {result.error}")
        _atomic_write_json(status_path, status)

    write_ensemble_debug_csv(all_debug_rows, exp_dir / "ensemble_debug.csv")

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
        "num_videos": len(selected_videos),
        "prediction_sets": [cfg.to_dict() for cfg in enabled_sets],
        "warnings": warnings,
        "submission_zip": str(exp_dir / "submission.zip") if args.make_submission else None,
    }
    _atomic_write_json(status_path, status)
    print(json.dumps(status["summary"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

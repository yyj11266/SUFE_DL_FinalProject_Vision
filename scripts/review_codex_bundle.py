"""Review a compact Codex bundle without expanding full experiment outputs."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.submission import validate_submission_zip


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and summarize a Drive-synced Codex review bundle.")
    parser.add_argument("--bundle-dir", required=True, help="Directory containing submission.zip, format_spec.json, and data_info.json.")
    parser.add_argument("--baseline-submission", help="Optional baseline submission.zip for pixel-difference summaries.")
    parser.add_argument("--output-json", help="Summary JSON path. Defaults to bundle/codex_review_summary.json.")
    parser.add_argument("--output-md", help="Markdown report path. Defaults to bundle/codex_review_report.md.")
    parser.add_argument("--large-delta-ratio", type=float, default=0.50)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _data_info_object_ids(data_info: dict[str, Any]) -> dict[str, set[int]]:
    ids: dict[str, set[int]] = defaultdict(set)
    for video in data_info.get("videos", []) or []:
        video_id = str(video.get("video_id", ""))
        for prompt in video.get("prompts", []) or []:
            for value in prompt.get("object_ids", []) or []:
                if int(value) > 0:
                    ids[video_id].add(int(value))
    return ids


def _mask_entries(format_spec: dict[str, Any]) -> list[dict[str, Any]]:
    entries = list(format_spec.get("expected_masks", []) or [])
    if entries:
        return sorted(entries, key=lambda item: (str(item.get("video_id", "")), int(item.get("frame_index", 0))))
    return []


def _video_id_from_path(path: str) -> str:
    parts = PurePosixPath(path).parts
    return parts[-2] if len(parts) >= 2 else ""


def _read_mask(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(zf.read(name))) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    return array


def _series_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"frames": 0}
    arr = np.asarray(values, dtype=np.float64)
    non_first = values[1:]
    zero_non_first = [index + 1 for index, value in enumerate(non_first) if int(value) == 0]
    deltas = np.abs(np.diff(arr)) if len(arr) > 1 else np.asarray([], dtype=np.float64)
    return {
        "frames": len(values),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "cv": float(arr.std() / arr.mean()) if arr.mean() > 0 else None,
        "zero_non_first_count": len(zero_non_first),
        "first_zero_non_first": zero_non_first[0] if zero_non_first else None,
        "max_delta_abs": int(deltas.max()) if len(deltas) else 0,
    }


def _summarize_submission(
    submission_zip: Path,
    format_spec: dict[str, Any],
    data_info: dict[str, Any],
    baseline_submission: Path | None,
    large_delta_ratio: float,
) -> dict[str, Any]:
    expected = _mask_entries(format_spec)
    expected_by_path = {str(item["relative_path"]): item for item in expected if item.get("relative_path")}
    known_ids = _data_info_object_ids(data_info)
    video_stats: dict[str, dict[str, Any]] = {}
    object_areas: dict[tuple[str, int], list[int]] = defaultdict(list)
    warnings: list[str] = []
    diff_summary = {
        "enabled": baseline_submission is not None,
        "frames_compared": 0,
        "changed_frames": 0,
        "mean_changed_fraction": None,
        "max_changed_fraction": 0.0,
        "top_changed_frames": [],
    }
    changed_fractions: list[float] = []

    baseline_zf: zipfile.ZipFile | None = zipfile.ZipFile(baseline_submission, "r") if baseline_submission else None
    try:
        with zipfile.ZipFile(submission_zip, "r") as zf:
            names = sorted(expected_by_path) if expected_by_path else sorted(
                name for name in zf.namelist() if PurePosixPath(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            for order, name in enumerate(names):
                mask = _read_mask(zf, name)
                item = expected_by_path.get(name, {})
                video_id = str(item.get("video_id") or _video_id_from_path(name))
                frame_index = int(item.get("frame_index", order))
                positive = mask > 0
                present = sorted(int(value) for value in np.unique(mask) if int(value) > 0)
                stats = video_stats.setdefault(
                    video_id,
                    {
                        "frames": 0,
                        "non_first_empty_frames": [],
                        "unexpected_object_ids": set(),
                        "foreground_fractions": [],
                    },
                )
                stats["frames"] += 1
                stats["foreground_fractions"].append(float(positive.mean()))
                if frame_index > 0 and int(positive.sum()) == 0:
                    stats["non_first_empty_frames"].append(frame_index)
                illegal = set(present) - known_ids.get(video_id, set())
                if illegal and known_ids.get(video_id):
                    stats["unexpected_object_ids"].update(illegal)
                for object_id in sorted(known_ids.get(video_id, set()) or set(present)):
                    object_areas[(video_id, int(object_id))].append(int((mask == int(object_id)).sum()))

                if baseline_zf is not None and name in baseline_zf.namelist():
                    baseline = _read_mask(baseline_zf, name)
                    if baseline.shape == mask.shape:
                        changed = float((baseline != mask).mean())
                        changed_fractions.append(changed)
                        diff_summary["frames_compared"] += 1
                        if changed > 0:
                            diff_summary["changed_frames"] += 1
                            diff_summary["max_changed_fraction"] = max(float(diff_summary["max_changed_fraction"]), changed)
                            diff_summary["top_changed_frames"].append(
                                {"relative_path": name, "video_id": video_id, "frame_index": frame_index, "changed_fraction": changed}
                            )
    finally:
        if baseline_zf is not None:
            baseline_zf.close()

    videos: dict[str, Any] = {}
    for video_id, item in sorted(video_stats.items()):
        fg = item["foreground_fractions"]
        non_first_empty = item["non_first_empty_frames"]
        unexpected = sorted(int(value) for value in item["unexpected_object_ids"])
        if non_first_empty:
            warnings.append(f"{video_id}: {len(non_first_empty)} non-first empty frame(s); first={non_first_empty[:5]}")
        if unexpected:
            warnings.append(f"{video_id}: unexpected object ids {unexpected}")
        videos[video_id] = {
            "frames": int(item["frames"]),
            "min_foreground_fraction": min(fg) if fg else None,
            "max_foreground_fraction": max(fg) if fg else None,
            "mean_foreground_fraction": float(np.mean(fg)) if fg else None,
            "non_first_empty_frame_count": len(non_first_empty),
            "first_non_first_empty_frames": non_first_empty[:10],
            "unexpected_object_ids": unexpected,
        }

    objects: dict[str, Any] = {}
    for (video_id, object_id), values in sorted(object_areas.items()):
        stats = _series_stats(values)
        if stats.get("zero_non_first_count"):
            warnings.append(
                f"{video_id} obj{object_id}: {stats['zero_non_first_count']} non-first zero-area frame(s); "
                f"first={stats['first_zero_non_first']}"
            )
        mean = float(stats.get("mean") or 0.0)
        if mean > 0 and float(stats.get("max_delta_abs") or 0) / mean > large_delta_ratio:
            warnings.append(
                f"{video_id} obj{object_id}: max area delta {stats['max_delta_abs']} exceeds "
                f"{large_delta_ratio:.2f}x mean area"
            )
        objects[f"{video_id}:{object_id}"] = stats

    if changed_fractions:
        diff_summary["mean_changed_fraction"] = float(np.mean(changed_fractions))
        diff_summary["top_changed_frames"] = sorted(
            diff_summary["top_changed_frames"],
            key=lambda item: float(item["changed_fraction"]),
            reverse=True,
        )[:20]

    return {
        "submission_zip": str(submission_zip),
        "videos": videos,
        "objects": objects,
        "diff_vs_baseline": diff_summary,
        "warnings": warnings,
    }


def _write_markdown(path: Path, validation: dict[str, Any], summary: dict[str, Any]) -> None:
    warnings = summary.get("warnings", [])
    lines = [
        "# Codex Review Report",
        "",
        f"- Submission: `{summary['submission_zip']}`",
        f"- Validation passed: `{validation.get('passed')}`",
        f"- Videos reviewed: `{len(summary.get('videos', {}))}`",
        f"- Objects summarized: `{len(summary.get('objects', {}))}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Validation",
        "",
        "```json",
        json.dumps(validation, indent=2, ensure_ascii=True)[:12000],
        "```",
        "",
        "## Highest-Signal Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings[:80])
    else:
        lines.append("- No proxy warnings found.")
    diff = summary.get("diff_vs_baseline", {})
    if diff.get("enabled"):
        lines.extend(
            [
                "",
                "## Baseline Difference",
                "",
                f"- Frames compared: `{diff.get('frames_compared')}`",
                f"- Changed frames: `{diff.get('changed_frames')}`",
                f"- Mean changed fraction: `{diff.get('mean_changed_fraction')}`",
                f"- Max changed fraction: `{diff.get('max_changed_fraction')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    submission_zip = bundle_dir / "submission.zip"
    format_spec_path = bundle_dir / "format_spec.json"
    data_info_path = bundle_dir / "data_info.json"
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else bundle_dir / "codex_review_summary.json"
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else bundle_dir / "codex_review_report.md"
    baseline_submission = Path(args.baseline_submission).expanduser().resolve() if args.baseline_submission else None

    if not submission_zip.exists():
        raise FileNotFoundError(f"Missing submission.zip: {submission_zip}")
    if not format_spec_path.exists():
        raise FileNotFoundError(f"Missing format_spec.json: {format_spec_path}")
    if not data_info_path.exists():
        raise FileNotFoundError(f"Missing data_info.json: {data_info_path}")

    format_spec = _load_json(format_spec_path)
    data_info = _load_json(data_info_path)
    validation = validate_submission_zip(submission_zip, format_spec, data_info)
    summary = _summarize_submission(submission_zip, format_spec, data_info, baseline_submission, args.large_delta_ratio)
    payload = {"validation": validation, "summary": summary}
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    _write_markdown(output_md, validation, summary)
    print(json.dumps({"summary_json": str(output_json), "report_md": str(output_md), "passed": validation.get("passed")}, indent=2))
    return 0 if validation.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

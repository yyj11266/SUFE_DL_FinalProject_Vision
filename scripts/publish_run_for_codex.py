"""Publish a compact Codex review bundle from a full local experiment run.

The full experiment directory may contain tens of thousands of masks, overlays,
cache files, and anchor reruns. This publisher keeps those on the fast local
runtime disk and writes only a small, reviewable bundle to Google Drive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


KEY_FILES = (
    "submission.zip",
    "sanity_check.json",
    "data_info.json",
    "format_spec.json",
    "run_manifest.json",
)

PALETTE = (
    (244, 67, 54),
    (33, 150, 243),
    (76, 175, 80),
    (255, 193, 7),
    (156, 39, 176),
    (0, 188, 212),
    (255, 112, 67),
    (139, 195, 74),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a compact Drive-synced Codex review bundle.")
    parser.add_argument("--exp-dir", required=True, help="Full local experiment directory, e.g. /content/sufe_runs/EXP.")
    parser.add_argument("--publish-dir", required=True, help="Small output directory to sync through Drive.")
    parser.add_argument("--data-root", help="Extracted dataset root used to render frame/mask previews.")
    parser.add_argument("--baseline-exp", help="Optional baseline experiment directory recorded in the manifest.")
    parser.add_argument("--archive-dir", help="Optional directory for a single-file full experiment archive.")
    parser.add_argument("--make-full-archive", action="store_true", help="Write EXP.full.tar to --archive-dir.")
    parser.add_argument("--make-review-zip", action="store_true", help="Also zip the compact review bundle.")
    parser.add_argument("--replace", action="store_true", help="Replace --publish-dir if it already exists.")
    parser.add_argument("--preview-frames-per-video", type=int, default=7)
    parser.add_argument("--suspicious-frames-per-video", type=int, default=4)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL)
            .strip()
            or None
        )
    except Exception:
        return None


def _copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
    }


def _copy_logs(exp_dir: Path, publish_dir: Path) -> list[str]:
    src = exp_dir / "logs"
    if not src.exists():
        return []
    dst = publish_dir / "logs"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return [str(path.relative_to(publish_dir)) for path in sorted(dst.rglob("*")) if path.is_file()]


def _frame_lookup(data_info: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for video in data_info.get("videos", []) or []:
        video_id = str(video.get("video_id", ""))
        frames: dict[str, dict[str, Any]] = {}
        for frame in video.get("frames", []) or []:
            frames[str(frame.get("frame_stem", ""))] = frame
        if video_id:
            lookup[video_id] = frames
    return lookup


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _colorize_mask(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:, :] = (18, 18, 18)
    for index, object_id in enumerate(sorted(int(value) for value in np.unique(mask) if value > 0)):
        rgb[mask == object_id] = PALETTE[index % len(PALETTE)]
    return Image.fromarray(rgb, mode="RGB")


def _overlay(frame_path: Path | None, mask: np.ndarray) -> Image.Image:
    if frame_path and frame_path.exists():
        with Image.open(frame_path) as image:
            base = image.convert("RGB")
        if base.size != (mask.shape[1], mask.shape[0]):
            mask = np.asarray(Image.fromarray(mask).resize(base.size, Image.Resampling.NEAREST), dtype=np.uint8)
        arr = np.asarray(base).copy()
        for index, object_id in enumerate(sorted(int(value) for value in np.unique(mask) if value > 0)):
            color = np.asarray(PALETTE[index % len(PALETTE)], dtype=np.uint8)
            region = mask == object_id
            arr[region] = (0.55 * arr[region] + 0.45 * color).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    return _colorize_mask(mask)


def _tile(image: Image.Image, label: str, size: tuple[int, int] = (360, 240)) -> Image.Image:
    image = image.copy()
    image.thumbnail((size[0], size[1] - 26))
    canvas = Image.new("RGB", size, (24, 24, 24))
    canvas.paste(image, ((size[0] - image.width) // 2, 26 + (size[1] - 26 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size[0], 24), fill=(0, 0, 0))
    draw.text((6, 5), label[:52], fill=(235, 235, 235))
    return canvas


def _mask_stats(mask_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_area: int | None = None
    for order, path in enumerate(mask_paths):
        mask = _load_mask(path)
        area = int((mask > 0).sum())
        rows.append(
            {
                "order": order,
                "frame_stem": path.stem,
                "path": str(path),
                "foreground_pixels": area,
                "foreground_fraction": float((mask > 0).mean()),
                "object_ids_present": sorted(int(value) for value in np.unique(mask) if value > 0),
                "area_delta_abs": 0 if previous_area is None else abs(area - previous_area),
            }
        )
        previous_area = area
    return rows


def _preview_orders(rows: list[dict[str, Any]], frame_budget: int, suspicious_budget: int) -> list[int]:
    if not rows:
        return []
    count = len(rows)
    keep = {0, count // 2, count - 1}
    suspicious = sorted(rows[1:], key=lambda item: int(item["area_delta_abs"]), reverse=True)[:suspicious_budget]
    keep.update(int(item["order"]) for item in suspicious)
    return sorted(order for order in keep if 0 <= order < count)[:frame_budget]


def _render_sheet(
    video_id: str,
    mask_paths: list[Path],
    selected_orders: list[int],
    frames_by_stem: dict[str, dict[str, Any]],
    data_root: Path | None,
    output_path: Path,
) -> str | None:
    if not selected_orders:
        return None
    tiles: list[Image.Image] = []
    for order in selected_orders:
        mask_path = mask_paths[order]
        frame = frames_by_stem.get(mask_path.stem, {})
        frame_path = data_root / str(frame.get("relative_path")) if data_root and frame.get("relative_path") else None
        mask = _load_mask(mask_path)
        label = f"{video_id} {mask_path.stem} fg={(mask > 0).mean():.3f}"
        tiles.append(_tile(_overlay(frame_path, mask), label))
    sheet = Image.new("RGB", (360 * len(tiles), 240), (24, 24, 24))
    for index, item in enumerate(tiles):
        sheet.paste(item, (360 * index, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)
    return str(output_path)


def _build_previews(exp_dir: Path, publish_dir: Path, data_root: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    masks_root = exp_dir / "masks"
    if not masks_root.exists():
        return {"status": "skipped", "reason": "masks directory not found"}
    data_info = _load_json(exp_dir / "data_info.json")
    frames = _frame_lookup(data_info)
    preview_root = publish_dir / "previews" / "contact_sheets"
    videos: dict[str, Any] = {}
    for video_dir in sorted(path for path in masks_root.iterdir() if path.is_dir()):
        mask_paths = sorted(video_dir.glob("*.png"))
        rows = _mask_stats(mask_paths)
        orders = _preview_orders(rows, int(args.preview_frames_per_video), int(args.suspicious_frames_per_video))
        sheet = _render_sheet(video_dir.name, mask_paths, orders, frames.get(video_dir.name, {}), data_root, preview_root / f"{video_dir.name}.jpg")
        videos[video_dir.name] = {
            "frame_count": len(mask_paths),
            "selected_frame_stems": [mask_paths[order].stem for order in orders],
            "contact_sheet": str(Path(sheet).relative_to(publish_dir)) if sheet else None,
            "max_area_delta_abs": max((int(row["area_delta_abs"]) for row in rows), default=0),
            "empty_non_first_frames": [
                row["frame_stem"] for row in rows[1:] if int(row["foreground_pixels"]) == 0
            ][:20],
        }
    payload = {"status": "ready", "videos": videos}
    _write_json(publish_dir / "previews" / "preview_index.json", payload)
    return payload


def _make_full_archive(exp_dir: Path, archive_dir: Path) -> dict[str, Any]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{exp_dir.name}.full.tar"
    with tarfile.open(archive_path, "w") as tar:
        tar.add(exp_dir, arcname=exp_dir.name)
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256(archive_path),
    }


def _make_review_zip(publish_dir: Path) -> dict[str, Any]:
    zip_path = publish_dir / "codex_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(publish_dir.rglob("*")):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(publish_dir).as_posix())
    return {
        "path": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "sha256": _sha256(zip_path),
    }


def _write_readme(publish_dir: Path) -> None:
    text = f"""# Codex Review Bundle

This directory is the Drive-synced review surface for one SUFE VOS experiment.
It intentionally omits full masks/overlays/cache trees. Use `submission.zip`
plus `format_spec.json` and `data_info.json` as the authoritative prediction
artifact for validation and local review.

Suggested local review command:

```bash
python scripts/review_codex_bundle.py --bundle-dir {publish_dir}
```
"""
    (publish_dir / "codex_review_readme.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    exp_dir = Path(args.exp_dir).expanduser().resolve()
    publish_dir = Path(args.publish_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
    if publish_dir.exists() and any(publish_dir.iterdir()):
        if not args.replace:
            raise FileExistsError(f"Publish directory already has files; pass --replace to overwrite: {publish_dir}")
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Any] = {}
    for name in KEY_FILES:
        info = _copy_if_exists(exp_dir / name, publish_dir / name)
        if info:
            copied[name] = info
    copied_logs = _copy_logs(exp_dir, publish_dir)
    preview_info = _build_previews(exp_dir, publish_dir, data_root, args)

    archive_info = None
    if args.make_full_archive:
        if not args.archive_dir:
            raise ValueError("--make-full-archive requires --archive-dir")
        archive_info = _make_full_archive(exp_dir, Path(args.archive_dir).expanduser().resolve())

    manifest = {
        "version": 1,
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": exp_dir.name,
        "source_exp_dir": str(exp_dir),
        "publish_dir": str(publish_dir),
        "baseline_exp": str(Path(args.baseline_exp).expanduser().resolve()) if args.baseline_exp else None,
        "project_root": str(PROJECT_ROOT),
        "git_sha": _git_sha(),
        "copied_files": copied,
        "copied_logs": copied_logs,
        "previews": preview_info,
        "full_archive": archive_info,
        "review_zip": None,
        "notes": [
            "Full frame masks remain authoritative inside submission.zip.",
            "Previews are diagnostic samples selected from first/middle/last and large area-change frames.",
            "Leaderboard scores are not inferred by this publisher.",
        ],
    }
    _write_json(publish_dir / "artifact_manifest.json", manifest)
    _write_readme(publish_dir)
    if args.make_review_zip:
        manifest["review_zip"] = _make_review_zip(publish_dir)
        _write_json(publish_dir / "artifact_manifest.json", manifest)
    print(json.dumps({"publish_dir": str(publish_dir), "manifest": str(publish_dir / "artifact_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

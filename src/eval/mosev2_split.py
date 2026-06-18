"""Create a deterministic stratified MOSEv2 calibration/holdout split."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_key(path: Path | str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _find_named_root(root: Path, name: str) -> Path:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = sorted((path for path in root.rglob(name) if path.is_dir()), key=lambda path: len(path.parts))
    if not matches:
        raise FileNotFoundError(f"Could not find {name}/ under {root}")
    return matches[0]


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def _indexed_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = np.any(array[..., :3] > 0, axis=-1).astype(np.uint8)
    return array


def _object_ids(mask: np.ndarray) -> list[int]:
    positives = sorted(int(value) for value in np.unique(mask) if int(value) > 0)
    if set(int(value) for value in np.unique(mask)).issubset({0, 1, 255}) and positives:
        return [1]
    return [value for value in positives if value != 255]


@dataclass(slots=True)
class MoseVideoStats:
    video_id: str
    frame_count: int
    annotated_frame_count: int
    object_count: int
    min_initial_area_ratio: float
    median_initial_area_ratio: float
    mean_disappearance_fraction: float
    max_disappearance_fraction: float
    area_bin: int = 0
    length_bin: int = 0
    disappearance_bin: int = 0
    object_count_bin: str = "1"
    stratum: str = ""


def inspect_mosev2(root: str | Path) -> list[MoseVideoStats]:
    """Measure split features from a standard JPEGImages/Annotations tree."""

    dataset_root = Path(root).expanduser().resolve()
    frames_root = _find_named_root(dataset_root, "JPEGImages")
    annotations_root = _find_named_root(dataset_root, "Annotations")
    video_ids = sorted(
        set(path.name for path in frames_root.iterdir() if path.is_dir())
        & set(path.name for path in annotations_root.iterdir() if path.is_dir())
    )
    if not video_ids:
        raise RuntimeError("No videos shared by JPEGImages and Annotations")

    stats: list[MoseVideoStats] = []
    for video_id in video_ids:
        frame_paths = _image_files(frames_root / video_id)
        annotation_paths = _image_files(annotations_root / video_id)
        if not frame_paths or not annotation_paths:
            continue
        first = _indexed_mask(annotation_paths[0])
        ids = _object_ids(first)
        if not ids:
            continue
        pixel_count = float(first.shape[0] * first.shape[1])
        areas = [float((first > 0).mean())] if ids == [1] and 255 in np.unique(first) else [float((first == object_id).sum() / pixel_count) for object_id in ids]
        absent_counts = {object_id: 0 for object_id in ids}
        for path in annotation_paths:
            mask = _indexed_mask(path)
            for object_id in ids:
                present = bool(np.any(mask > 0)) if ids == [1] and 255 in np.unique(first) else bool(np.any(mask == object_id))
                absent_counts[object_id] += int(not present)
        fractions = [absent_counts[object_id] / len(annotation_paths) for object_id in ids]
        stats.append(
            MoseVideoStats(
                video_id=video_id,
                frame_count=len(frame_paths),
                annotated_frame_count=len(annotation_paths),
                object_count=len(ids),
                min_initial_area_ratio=float(min(areas)),
                median_initial_area_ratio=float(np.median(areas)),
                mean_disappearance_fraction=float(np.mean(fractions)),
                max_disappearance_fraction=float(max(fractions)),
            )
        )
    if not stats:
        raise RuntimeError("No valid MOSEv2 videos were found")
    return stats


def _rank_bins(values: list[float], bins: int = 4) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    labels = [0] * len(values)
    for rank, index in enumerate(order):
        labels[index] = min(bins - 1, rank * bins // max(1, len(values)))
    return labels


def _assign_strata(stats: list[MoseVideoStats]) -> None:
    area_bins = _rank_bins([item.min_initial_area_ratio for item in stats])
    length_bins = _rank_bins([float(item.frame_count) for item in stats])
    disappearance_bins = _rank_bins([item.max_disappearance_fraction for item in stats])
    for index, item in enumerate(stats):
        item.area_bin = area_bins[index]
        item.length_bin = length_bins[index]
        item.disappearance_bin = disappearance_bins[index]
        item.object_count_bin = "1" if item.object_count == 1 else "2" if item.object_count == 2 else "3+"
        item.stratum = (
            f"area{item.area_bin}|objects{item.object_count_bin}|"
            f"length{item.length_bin}|disappear{item.disappearance_bin}"
        )


def _round_robin_sample(stats: list[MoseVideoStats], total: int, seed: int) -> list[MoseVideoStats]:
    grouped: dict[str, list[MoseVideoStats]] = defaultdict(list)
    for item in stats:
        grouped[item.stratum].append(item)
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    strata = sorted(grouped)
    rng.shuffle(strata)
    selected: list[MoseVideoStats] = []
    while len(selected) < total:
        progress = False
        for stratum in strata:
            if grouped[stratum] and len(selected) < total:
                selected.append(grouped[stratum].pop())
                progress = True
        if not progress:
            break
    return selected


def _distribution(items: Iterable[MoseVideoStats]) -> dict[str, Any]:
    values = list(items)
    return {
        "videos": len(values),
        "objects": sum(item.object_count for item in values),
        "single_object_videos": sum(item.object_count == 1 for item in values),
        "multi_object_videos": sum(item.object_count > 1 for item in values),
        "small_initial_area_lt_1pct": sum(item.min_initial_area_ratio < 0.01 for item in values),
        "disappearance_present": sum(item.max_disappearance_fraction > 0 for item in values),
        "mean_frame_count": float(np.mean([item.frame_count for item in values])) if values else 0.0,
        "strata": dict(sorted(Counter(item.stratum for item in values).items())),
    }


def build_mosev2_split(
    root: str | Path,
    *,
    seed: int = 2026,
    total: int = 80,
    calibration_size: int = 40,
    allow_smaller: bool = False,
) -> dict[str, Any]:
    """Return a fixed stratified calibration/holdout manifest."""

    stats = inspect_mosev2(root)
    if len(stats) < total:
        if not allow_smaller:
            raise RuntimeError(f"Requested {total} videos but only {len(stats)} valid videos were found")
        total = len(stats)
        calibration_size = total // 2
    if calibration_size <= 0 or calibration_size >= total:
        raise ValueError("calibration_size must be between 1 and total-1")
    _assign_strata(stats)
    selected = _round_robin_sample(stats, total, seed)

    calibration: list[MoseVideoStats] = []
    holdout: list[MoseVideoStats] = []
    by_stratum: dict[str, list[MoseVideoStats]] = defaultdict(list)
    for item in selected:
        by_stratum[item.stratum].append(item)
    rng = random.Random(seed + 1)
    for stratum in sorted(by_stratum):
        group = by_stratum[stratum]
        rng.shuffle(group)
        for item in group:
            target_calibration = len(calibration) < calibration_size
            target_holdout = len(holdout) < total - calibration_size
            if target_calibration and target_holdout:
                destination = calibration if len(calibration) <= len(holdout) else holdout
            else:
                destination = calibration if target_calibration else holdout
            destination.append(item)

    calibration = sorted(calibration, key=lambda item: item.video_id)
    holdout = sorted(holdout, key=lambda item: item.video_id)
    return {
        "schema_version": 1,
        "dataset_root": str(Path(root).expanduser().resolve()),
        "seed": seed,
        "selection_policy": "rank-quartile composite strata with deterministic round-robin sampling",
        "threshold_policy": "tune only on calibration; confirm once on holdout",
        "calibration": [item.video_id for item in calibration],
        "holdout": [item.video_id for item in holdout],
        "distribution": {
            "calibration": _distribution(calibration),
            "holdout": _distribution(holdout),
        },
        "videos": [asdict(item) for item in sorted(selected, key=lambda item: item.video_id)],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the frozen MOSEv2 calibration/holdout split.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--total", type=int, default=80)
    parser.add_argument("--calibration-size", type=int, default=40)
    parser.add_argument("--allow-smaller", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_mosev2_split(
        args.root,
        seed=args.seed,
        total=args.total,
        calibration_size=args.calibration_size,
        allow_smaller=args.allow_smaller,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest["distribution"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect SUFE video object segmentation data layouts.

The helpers in this module are intentionally conservative: they collect
structure facts without assuming the leaderboard archive uses one fixed
directory schema. The returned dataclasses are JSON-serializable through
``to_dict`` and are also exposed through a small argparse CLI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
PROMPT_EXTENSIONS = {".json", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROMPT_STEMS = {
    "prompt",
    "prompts",
    "initial_prompt",
    "initial_prompts",
    "first_frame_prompt",
    "first_frame_prompts",
    "annotation",
    "annotations",
}
MASK_DIR_NAMES = {"annotations", "annotation", "masks", "mask", "labels", "label"}
FRAME_DIR_NAMES = {"jpegimages", "images", "imgs", "frames", "video", "videos"}
IGNORED_DIR_NAMES = {"__macosx", ".ipynb_checkpoints"}


@dataclass(slots=True)
class FrameInfo:
    """Description of one image frame discovered in the dataset."""

    video_id: str
    frame_index: int
    frame_stem: str
    relative_path: str
    width: int
    height: int
    source_type: str = "image"


@dataclass(slots=True)
class PromptInfo:
    """Description of an initial prompt or first-frame annotation."""

    video_id: str
    prompt_type: str
    relative_path: str
    frame_index: int | None = None
    object_ids: list[int] = field(default_factory=list)
    mask_encoding: str = "unknown"
    width: int | None = None
    height: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoInfo:
    """Description of a video sequence, represented either by frames or a video file."""

    video_id: str
    source_type: str
    relative_path: str
    frame_count: int
    width: int | None
    height: int | None
    fps: float | None = None
    frames: list[FrameInfo] = field(default_factory=list)
    prompts: list[PromptInfo] = field(default_factory=list)


@dataclass(slots=True)
class DataInfo:
    """Full inspection result for an extracted SUFE dataset."""

    root: str
    videos: list[VideoInfo]
    prompt_format: dict[str, Any]
    scan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the inspection result."""

        return asdict(self)


def _as_path(root: str | Path) -> Path:
    """Return a resolved path and fail early when it does not exist."""

    path = Path(root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _relative(path: Path, root: Path) -> str:
    """Return a POSIX-style relative path."""

    return path.relative_to(root).as_posix()


def _natural_key(path: Path | str) -> list[int | str]:
    """Sort strings with embedded numbers in numeric order."""

    text = path.as_posix() if isinstance(path, Path) else str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _is_ignored_artifact(path: Path) -> bool:
    """Return whether a file is an OS/editor artifact rather than dataset content."""

    return any(_is_ignored_path_part(part) for part in path.parts)


def _is_ignored_path_part(part: str) -> bool:
    """Return whether a path component should be skipped while scanning data."""

    return part.lower() in IGNORED_DIR_NAMES or part.startswith(".") or part.startswith("._")


def _frame_index_from_stem(stem: str, fallback: int) -> int:
    """Extract a frame index from a filename stem, falling back to sequence order."""

    if stem.isdigit():
        return int(stem)
    numbers = re.findall(r"\d+", stem)
    return int(numbers[-1]) if numbers else fallback


def _image_size(path: Path) -> tuple[int, int]:
    """Read image size as ``(width, height)`` without loading more than needed."""

    with Image.open(path) as image:
        return image.size


def _is_under_named_dir(path: Path, names: set[str]) -> bool:
    """Return whether any path component matches a normalized directory name."""

    return any(part.lower() in names for part in path.parts)


def _list_files(root: Path) -> list[Path]:
    """List files recursively in stable order."""

    files: list[Path] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = [dirname for dirname in dirs if not _is_ignored_path_part(dirname)]
        current_path = Path(current)
        for name in names:
            path = current_path / name
            if path.is_file() and not _is_ignored_artifact(path):
                files.append(path)
    return sorted(files, key=_natural_key)


def recursively_scan(root: str | Path) -> dict[str, Any]:
    """Recursively scan a dataset root and summarize candidate file types.

    Args:
        root: Extracted dataset root.

    Returns:
        A JSON-serializable summary with file counts and representative paths.
    """

    root_path = _as_path(root)
    files = _list_files(root_path)
    suffix_counts = Counter((path.suffix.lower() or "<none>") for path in files)

    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    video_files = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    prompt_files = [
        path
        for path in files
        if path.suffix.lower() in PROMPT_EXTENSIONS
        and (path.stem.lower() in PROMPT_STEMS or _is_under_named_dir(path, MASK_DIR_NAMES))
    ]
    sample_files = [
        path
        for path in files
        if "sample" in path.name.lower() and "submission" in path.name.lower()
    ]

    def examples(paths: Iterable[Path], limit: int = 30) -> list[str]:
        return [_relative(path, root_path) for path in list(paths)[:limit]]

    return {
        "root": str(root_path),
        "num_files": len(files),
        "suffix_counts": dict(suffix_counts.most_common()),
        "num_image_files": len(image_files),
        "num_video_files": len(video_files),
        "num_prompt_files": len(prompt_files),
        "num_sample_submission_files": len(sample_files),
        "image_examples": examples(image_files),
        "video_examples": examples(video_files),
        "prompt_examples": examples(prompt_files),
        "sample_submission_examples": examples(sample_files),
        "top_level_entries": sorted((path.name for path in root_path.iterdir()), key=str.lower),
    }


def _video_id_from_frame_path(path: Path, root: Path) -> str | None:
    """Infer video id for a frame path from known SUFE/VOS layouts."""

    rel_parts = path.relative_to(root).parts
    lower_parts = [part.lower() for part in rel_parts]

    if "annotations" in lower_parts or "masks" in lower_parts:
        return None

    if "jpegimages" in lower_parts:
        idx = lower_parts.index("jpegimages")
        if idx + 1 < len(rel_parts) - 1:
            return rel_parts[idx + 1]

    if "videos" in lower_parts:
        idx = lower_parts.index("videos")
        if idx + 1 < len(rel_parts) - 1:
            return rel_parts[idx + 1]

    if "frames" in lower_parts:
        idx = lower_parts.index("frames")
        if idx - 1 >= 0:
            return rel_parts[idx - 1]

    if _is_under_named_dir(path.relative_to(root), MASK_DIR_NAMES):
        return None

    parent = path.parent
    image_siblings = [
        p
        for p in parent.iterdir()
        if p.is_file() and not _is_ignored_artifact(p) and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(image_siblings) >= 2:
        return parent.name
    return None


def detect_frames(root: str | Path) -> dict[str, list[FrameInfo]]:
    """Detect frame-image sequences grouped by video id.

    Supported layouts include:
        ``JPEGImages/{video}/00000.jpg``
        ``videos/{video_id}/*.jpg``
        ``{video_id}/frames/*.jpg``
        fallback directories containing multiple image files
    """

    root_path = _as_path(root)
    grouped_paths: dict[str, list[Path]] = defaultdict(list)

    for path in _list_files(root_path):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        video_id = _video_id_from_frame_path(path, root_path)
        if video_id:
            grouped_paths[video_id].append(path)

    grouped_frames: dict[str, list[FrameInfo]] = {}
    for video_id, paths in sorted(grouped_paths.items(), key=lambda item: item[0].lower()):
        sorted_paths = sorted(paths, key=_natural_key)
        first_width, first_height = _image_size(sorted_paths[0])
        frames: list[FrameInfo] = []
        for order, path in enumerate(sorted_paths):
            frames.append(
                FrameInfo(
                    video_id=video_id,
                    frame_index=_frame_index_from_stem(path.stem, order),
                    frame_stem=path.stem,
                    relative_path=_relative(path, root_path),
                    width=first_width,
                    height=first_height,
                )
            )
        grouped_frames[video_id] = frames
    return grouped_frames


def _inspect_video_file(path: Path) -> tuple[int, int | None, int | None, float | None]:
    """Inspect a video file with OpenCV and return count, width, height, and fps."""

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, None, None, None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
    cap.release()
    return frame_count, width, height, fps


def detect_video_dirs(root: str | Path) -> dict[str, VideoInfo]:
    """Detect image-sequence video directories and standalone video files."""

    root_path = _as_path(root)
    frames_by_video = detect_frames(root_path)
    videos: dict[str, VideoInfo] = {}

    for video_id, frames in frames_by_video.items():
        first = frames[0] if frames else None
        common_parent = Path(first.relative_path).parent.as_posix() if first else video_id
        videos[video_id] = VideoInfo(
            video_id=video_id,
            source_type="frames",
            relative_path=common_parent,
            frame_count=len(frames),
            width=first.width if first else None,
            height=first.height if first else None,
            frames=frames,
        )

    for path in _list_files(root_path):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        video_id = path.stem
        frame_count, width, height, fps = _inspect_video_file(path)
        videos.setdefault(
            video_id,
            VideoInfo(
                video_id=video_id,
                source_type="video_file",
                relative_path=_relative(path, root_path),
                frame_count=frame_count,
                width=width,
                height=height,
                fps=fps,
            ),
        )

    return dict(sorted(videos.items(), key=lambda item: item[0].lower()))


def _mask_prompt_info(path: Path, root: Path, video_id: str, frame_index: int | None) -> PromptInfo:
    """Build prompt metadata for a mask prompt file."""

    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        nonzero = np.any(array[..., :3] > 0, axis=-1)
        unique_values = np.unique(nonzero.astype(np.uint8))
    else:
        unique_values = np.unique(array)
    unique_ints = sorted(int(value) for value in unique_values.tolist())
    object_ids = [value for value in unique_ints if value > 0 and value != 255]
    if not object_ids and any(value > 0 for value in unique_ints):
        object_ids = [1]

    positive_values = [value for value in unique_ints if value > 0]
    if set(unique_ints).issubset({0, 1, 255}) and len(positive_values) <= 2:
        mask_encoding = "binary_png"
    else:
        mask_encoding = "indexed_png"

    height, width = array.shape[:2]
    return PromptInfo(
        video_id=video_id,
        prompt_type="mask",
        relative_path=_relative(path, root),
        frame_index=frame_index,
        object_ids=object_ids,
        mask_encoding=mask_encoding,
        width=width,
        height=height,
        details={"unique_values": unique_ints[:256]},
    )


def _classify_json_prompt(payload: Any) -> tuple[str, list[int], dict[str, Any]]:
    """Classify a JSON prompt payload as bbox, points, mask, mixed, or unknown."""

    text = json.dumps(payload).lower()
    has_box = any(key in text for key in ("bbox", "boxes", "box", "rectangle"))
    has_points = any(key in text for key in ("point", "points", "click", "clicks"))
    has_mask = any(key in text for key in ("mask", "rle", "segmentation"))
    object_ids: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in {"object_id", "obj_id", "id", "label", "category_id"}:
                    if isinstance(nested, int) and nested > 0:
                        object_ids.add(int(nested))
                    elif isinstance(nested, str) and nested.isdigit() and int(nested) > 0:
                        object_ids.add(int(nested))
                walk(nested)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)

    flags = [has_box, has_points, has_mask]
    if sum(flags) > 1:
        prompt_type = "mixed"
    elif has_box:
        prompt_type = "bbox"
    elif has_points:
        prompt_type = "points"
    elif has_mask:
        prompt_type = "mask"
    else:
        prompt_type = "unknown"
    return prompt_type, sorted(object_ids), {"top_level_type": type(payload).__name__}


def _json_prompt_infos(path: Path, root: Path) -> list[PromptInfo]:
    """Parse JSON prompts and return one or more prompt descriptions."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return [
            PromptInfo(
                video_id=path.parent.name,
                prompt_type="unknown",
                relative_path=_relative(path, root),
                details={"parse_error": "json_decode_failed"},
            )
        ]

    infos: list[PromptInfo] = []
    if isinstance(payload, dict):
        candidate_items = [
            (str(key), value)
            for key, value in payload.items()
            if isinstance(value, (dict, list)) and key not in {"bbox", "box", "points", "mask"}
        ]
        if candidate_items and path.stem.lower() in PROMPT_STEMS:
            for video_id, value in candidate_items:
                prompt_type, object_ids, details = _classify_json_prompt(value)
                infos.append(
                    PromptInfo(
                        video_id=video_id,
                        prompt_type=prompt_type,
                        relative_path=_relative(path, root),
                        object_ids=object_ids,
                        details=details,
                    )
                )
        if infos:
            return infos

    prompt_type, object_ids, details = _classify_json_prompt(payload)
    return [
        PromptInfo(
            video_id=path.parent.name,
            prompt_type=prompt_type,
            relative_path=_relative(path, root),
            object_ids=object_ids,
            details=details,
        )
    ]


def _text_prompt_info(path: Path, root: Path) -> PromptInfo:
    """Classify simple CSV/TXT prompt files by token shape."""

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="ignore")
    numbers = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(numbers) >= 4 and len(numbers) % 4 == 0:
        prompt_type = "bbox"
    elif len(numbers) >= 2:
        prompt_type = "points"
    else:
        prompt_type = "unknown"
    return PromptInfo(
        video_id=path.parent.name,
        prompt_type=prompt_type,
        relative_path=_relative(path, root),
        details={"num_numeric_tokens": len(numbers)},
    )


def _annotation_mask_prompts(root: Path) -> list[PromptInfo]:
    """Find first-frame masks under annotation-like directories."""

    paths_by_video: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in _list_files(root):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_parts = path.relative_to(root).parts
        lower_parts = [part.lower() for part in rel_parts]
        annotation_indices = [idx for idx, part in enumerate(lower_parts) if part in MASK_DIR_NAMES]
        if not annotation_indices:
            continue
        idx = annotation_indices[0]
        if idx + 1 >= len(rel_parts) - 1:
            continue
        video_id = rel_parts[idx + 1]
        frame_index = _frame_index_from_stem(path.stem, 0)
        paths_by_video[video_id].append((frame_index, path))

    prompts: list[PromptInfo] = []
    for video_id, indexed_paths in sorted(paths_by_video.items(), key=lambda item: item[0].lower()):
        frame_index, path = sorted(indexed_paths, key=lambda item: (item[0], _natural_key(item[1])))[0]
        prompts.append(_mask_prompt_info(path, root, video_id, frame_index))
    return prompts


def _named_prompt_files(root: Path) -> list[PromptInfo]:
    """Find prompt files named prompt.*, prompts.*, or first_frame_prompt.*."""

    prompts: list[PromptInfo] = []
    for path in _list_files(root):
        if path.suffix.lower() not in PROMPT_EXTENSIONS:
            continue
        if path.stem.lower() not in PROMPT_STEMS:
            continue
        if _is_under_named_dir(path.relative_to(root), MASK_DIR_NAMES) and path.suffix.lower() in IMAGE_EXTENSIONS:
            continue
        if path.suffix.lower() == ".json":
            prompts.extend(_json_prompt_infos(path, root))
        elif path.suffix.lower() in {".csv", ".txt"}:
            if path.suffix.lower() == ".csv":
                try:
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        list(csv.reader(handle))
                except UnicodeDecodeError:
                    pass
            prompts.append(_text_prompt_info(path, root))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            prompts.append(_mask_prompt_info(path, root, path.parent.name, None))
    return prompts


def detect_initial_prompts(root: str | Path) -> dict[str, list[PromptInfo]]:
    """Detect initial prompts grouped by video id.

    The detector supports first-frame masks in annotation directories, prompt image
    files, and lightweight JSON/CSV/TXT prompt descriptions.
    """

    root_path = _as_path(root)
    prompts = _annotation_mask_prompts(root_path) + _named_prompt_files(root_path)
    grouped: dict[str, list[PromptInfo]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for prompt in prompts:
        key = (prompt.video_id, prompt.relative_path, prompt.prompt_type)
        if key in seen:
            continue
        seen.add(key)
        grouped[prompt.video_id].append(prompt)
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def _summarize_prompt_format(prompts_by_video: dict[str, list[PromptInfo]]) -> dict[str, Any]:
    """Summarize detected prompt types and object-id conventions."""

    all_prompts = [prompt for prompts in prompts_by_video.values() for prompt in prompts]
    type_counts = Counter(prompt.prompt_type for prompt in all_prompts)
    encoding_counts = Counter(prompt.mask_encoding for prompt in all_prompts if prompt.mask_encoding != "unknown")
    all_object_ids = sorted({object_id for prompt in all_prompts for object_id in prompt.object_ids})
    frame_indices = [prompt.frame_index for prompt in all_prompts if prompt.frame_index is not None]

    if encoding_counts.get("indexed_png"):
        object_rule = "pixel_value_object_id"
    elif encoding_counts.get("binary_png"):
        object_rule = "foreground_is_1_or_255"
    elif any(prompt.prompt_type in {"bbox", "points", "mixed"} for prompt in all_prompts):
        object_rule = "prompt_file_object_id_if_present"
    else:
        object_rule = "unknown"

    return {
        "num_prompted_videos": len(prompts_by_video),
        "prompt_type_counts": dict(type_counts),
        "mask_encoding_counts": dict(encoding_counts),
        "object_ids": all_object_ids,
        "object_id_encoding_rule": object_rule,
        "first_frame_index_min": min(frame_indices) if frame_indices else None,
        "first_frame_index_max": max(frame_indices) if frame_indices else None,
    }


def detect_prompt_format(root: str | Path) -> dict[str, Any]:
    """Detect and summarize prompt types and object-id conventions."""

    return _summarize_prompt_format(detect_initial_prompts(root))


def inspect_dataset(root: str | Path) -> DataInfo:
    """Run the full SUFE dataset inspection pipeline."""

    root_path = _as_path(root)
    scan = recursively_scan(root_path)
    videos_by_id = detect_video_dirs(root_path)
    prompts_by_video = detect_initial_prompts(root_path)
    for video_id, prompts in prompts_by_video.items():
        if video_id in videos_by_id:
            videos_by_id[video_id].prompts = prompts
        else:
            videos_by_id[video_id] = VideoInfo(
                video_id=video_id,
                source_type="prompt_only",
                relative_path=Path(prompts[0].relative_path).parent.as_posix(),
                frame_count=0,
                width=prompts[0].width,
                height=prompts[0].height,
                prompts=prompts,
            )
    return DataInfo(
        root=str(root_path),
        videos=list(videos_by_id.values()),
        prompt_format=_summarize_prompt_format(prompts_by_video),
        scan=scan,
    )


def save_data_info(data_info: DataInfo, output: str | Path) -> None:
    """Write a ``DataInfo`` object to JSON."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data_info.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")


def load_data_info(path: str | Path) -> dict[str, Any]:
    """Load a previously saved data-info JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset inspection."""

    parser = argparse.ArgumentParser(description="Inspect an extracted SUFE VOS dataset.")
    parser.add_argument("--root", required=True, help="Extracted dataset root.")
    parser.add_argument("--output", required=True, help="Output data_info.json path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m src.data.inspect_sufe``."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    data_info = inspect_dataset(args.root)
    save_data_info(data_info, args.output)
    print(json.dumps(data_info.to_dict(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

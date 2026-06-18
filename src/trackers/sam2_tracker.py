"""SAM2.1 video tracking baseline for SUFE leaderboard submissions."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageColor


SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
CHECKPOINT_URLS = {
    "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2.1_hiera_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}
DEFAULT_CHECKPOINT_NAME = "sam2.1_hiera_large"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_RUNTIME_DEPENDENCIES = {
    "hydra": "hydra-core>=1.3.2",
    "iopath": "iopath>=0.1.10",
}


@dataclass(slots=True)
class PreparedFrame:
    """Mapping between one original frame and one SAM2 inference frame."""

    frame_index: int
    frame_stem: str
    inference_path: str
    original_path: str
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int


@dataclass(slots=True)
class ObjectPrompt:
    """SAM2 prompt bundle for one object in the first frame."""

    object_id: int
    mask: np.ndarray
    box: np.ndarray
    points: np.ndarray
    labels: np.ndarray
    original_box: list[int]


@dataclass(slots=True)
class Sam2VideoResult:
    """Result metadata for one SAM2 video run."""

    video_id: str
    status: str
    frame_count: int
    object_ids: list[int] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    overlay_paths: list[str] = field(default_factory=list)
    raw_logit_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _import_torch() -> Any:
    """Import torch lazily and return the module."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError("PyTorch is required for SAM2 inference but could not be imported.") from exc
    return torch


def _ensure_repo_importable(repo_path: Path) -> bool:
    """Add a cloned SAM2 repo to ``sys.path`` when it has a package tree."""

    if not (repo_path / "sam2" / "__init__.py").exists():
        return False
    repo_text = str(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    importlib.invalidate_caches()
    return importlib.util.find_spec("sam2") is not None


def _install_missing_sam2_runtime_dependencies() -> None:
    """Install lightweight SAM2 runtime dependencies when absent."""

    missing = [package for module, package in SAM2_RUNTIME_DEPENDENCIES.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as exc:
        raise RuntimeError(f"Failed to install required SAM2 runtime dependencies: {missing}") from exc
    importlib.invalidate_caches()


def install_or_check_sam2(repo_dir: str | Path) -> Path:
    """Install or verify the official SAM2 repository.

    The function first checks whether ``sam2`` is already importable. If not, it
    clones the official Meta repository into ``repo_dir`` and runs
    ``python -m pip install -e <repo_dir>``.
    """

    existing_spec = importlib.util.find_spec("sam2")
    if existing_spec and existing_spec.origin:
        _install_missing_sam2_runtime_dependencies()
        return Path(existing_spec.origin).resolve().parents[1]

    repo_path = Path(repo_dir).expanduser().resolve()
    if _ensure_repo_importable(repo_path):
        _install_missing_sam2_runtime_dependencies()
        return repo_path

    try:
        if repo_path.exists() and not (repo_path / "sam2" / "__init__.py").exists():
            shutil.rmtree(repo_path)
        if not repo_path.exists():
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            clone_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    subprocess.check_call(
                        ["git", "clone", "--depth", "1", "--filter=blob:none", SAM2_REPO_URL, str(repo_path)]
                    )
                    clone_error = None
                    break
                except Exception as exc:
                    clone_error = exc
                    if repo_path.exists():
                        shutil.rmtree(repo_path)
            if clone_error is not None:
                raise clone_error
        if not (repo_path / "sam2").exists():
            raise RuntimeError(f"Official SAM2 package directory is missing under {repo_path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_path)])
    except Exception as exc:
        raise RuntimeError(
            "Failed to install official SAM2 repository. "
            f"repo_dir={repo_path}, clone_url={SAM2_REPO_URL}, "
            f"command='{sys.executable} -m pip install -e {repo_path}'."
        ) from exc

    if _ensure_repo_importable(repo_path):
        _install_missing_sam2_runtime_dependencies()
        return repo_path

    installed_spec = importlib.util.find_spec("sam2")
    if not installed_spec:
        raise RuntimeError(
            "SAM2 installation completed but 'sam2' is still not importable. "
            f"Check Python environment and repo_dir={repo_path}."
        )
    return repo_path


def _checkpoint_filename(checkpoint_name: str) -> str:
    """Return a checkpoint filename for a known SAM2 checkpoint key."""

    if checkpoint_name in {"sam2.1_hiera_large", "sam2.1_hiera_l"}:
        return "sam2.1_hiera_large.pt"
    return checkpoint_name if checkpoint_name.endswith(".pt") else f"{checkpoint_name}.pt"


def download_sam2_checkpoint(
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    checkpoint_dir: str | Path | None = None,
) -> Path:
    """Download a SAM2 checkpoint if needed and return its local path."""

    checkpoint_candidate = Path(checkpoint_name).expanduser()
    if checkpoint_candidate.exists():
        if checkpoint_candidate.stat().st_size <= 0:
            raise RuntimeError(f"Checkpoint file is empty: {checkpoint_candidate}")
        return checkpoint_candidate.resolve()

    if checkpoint_name not in CHECKPOINT_URLS:
        raise ValueError(
            f"Unknown SAM2 checkpoint name {checkpoint_name!r}. "
            f"Known names: {sorted(CHECKPOINT_URLS)}. Pass a local .pt path instead."
        )

    target_dir = Path(checkpoint_dir or Path.cwd() / "checkpoints").expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / _checkpoint_filename(checkpoint_name)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    url = CHECKPOINT_URLS[checkpoint_name]
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Failed to download SAM2 checkpoint from {url} to {output_path}") from exc
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded checkpoint is missing or empty: {output_path}")
    return output_path


def build_sam2_video_predictor(
    checkpoint_path: str | Path,
    model_cfg: str,
    device: str,
    vos_optimized: bool = False,
) -> Any:
    """Build the official SAM2 video predictor with clear failure messages."""

    torch = _import_torch()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"SAM2 checkpoint is empty: {checkpoint}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SAM2 baseline requires CUDA, but torch.cuda.is_available() is False.")
    if Path(model_cfg).is_absolute() and not Path(model_cfg).exists():
        raise FileNotFoundError(f"SAM2 model config does not exist: {model_cfg}")

    try:
        from sam2.build_sam import build_sam2_video_predictor as official_builder
    except Exception as exc:
        raise RuntimeError(
            "Could not import official SAM2 builder after repository setup: "
            f"{type(exc).__name__}: {exc}. "
            "Ensure hydra-core, iopath, torch, and torchvision are installed."
        ) from exc

    kwargs: dict[str, Any] = {}
    try:
        signature = inspect.signature(official_builder)
        if "device" in signature.parameters:
            kwargs["device"] = device
        if "vos_optimized" in signature.parameters:
            kwargs["vos_optimized"] = vos_optimized
    except (TypeError, ValueError):
        pass

    try:
        predictor = official_builder(model_cfg, str(checkpoint), **kwargs)
        if hasattr(predictor, "to") and "device" not in kwargs:
            predictor = predictor.to(device)
        return predictor
    except Exception as exc:
        raise RuntimeError(
            "Failed to build SAM2 video predictor. "
            f"checkpoint={checkpoint}, model_cfg={model_cfg}, device={device}, "
            f"vos_optimized={vos_optimized}."
        ) from exc


def _natural_key(text: str) -> list[int | str]:
    """Sort text with embedded numbers in numeric order."""

    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def _video_id(video_info: Any) -> str:
    """Extract video id from a dataclass or dictionary."""

    return str(getattr(video_info, "video_id", None) or video_info["video_id"])


def _video_frames(video_info: Any) -> list[Any]:
    """Extract frame metadata from a dataclass or dictionary."""

    frames = getattr(video_info, "frames", None)
    if frames is None:
        frames = video_info.get("frames", [])
    return list(frames)


def _field(item: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from a dataclass-like object or a dictionary."""

    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, dict):
        return item.get(name, default)
    return default


def _resize_size(width: int, height: int, resize_long_side: int | None) -> tuple[int, int]:
    """Compute inference size for an optional long-side resize."""

    if not resize_long_side or resize_long_side <= 0:
        return width, height
    long_side = max(width, height)
    if long_side <= resize_long_side:
        return width, height
    scale = resize_long_side / float(long_side)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _write_rgb_frame(source: Path, target: Path, resize_long_side: int | None) -> tuple[int, int, int, int]:
    """Write an RGB JPEG inference frame and return original/inference dimensions."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        original_width, original_height = rgb.size
        inference_width, inference_height = _resize_size(original_width, original_height, resize_long_side)
        if (inference_width, inference_height) != rgb.size:
            rgb = rgb.resize((inference_width, inference_height), Image.Resampling.BILINEAR)
        rgb.save(target, quality=95)
    return original_width, original_height, inference_width, inference_height


def _prepare_frame_dir(video_info: Any, config: dict[str, Any]) -> list[PreparedFrame]:
    """Create a SAM2-friendly frame directory for one video."""

    data_root = Path(config["data_root"]).expanduser().resolve()
    cache_dir = Path(config["cache_dir"]).expanduser().resolve()
    video_id = _video_id(video_info)
    inference_dir = cache_dir / "frames" / video_id
    original_dir = cache_dir / "original_frames" / video_id
    inference_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)
    resize_long_side = int(config.get("resize_long_side") or 0)
    output_frame_stems = list(config.get("output_frame_stems") or [])

    prepared: list[PreparedFrame] = []
    frames = _video_frames(video_info)
    if frames:
        for order, frame in enumerate(sorted(frames, key=lambda item: _natural_key(str(_field(item, "relative_path", ""))))):
            source = data_root / str(_field(frame, "relative_path"))
            if not source.exists():
                raise FileNotFoundError(f"Frame file does not exist: {source}")
            frame_stem = output_frame_stems[order] if order < len(output_frame_stems) else str(_field(frame, "frame_stem", f"{order:05d}"))
            inference_path = inference_dir / f"{order:05d}.jpg"
            original_path = original_dir / f"{order:05d}.jpg"
            original_width, original_height, inference_width, inference_height = _write_rgb_frame(source, inference_path, resize_long_side)
            if not original_path.exists():
                _write_rgb_frame(source, original_path, None)
            prepared.append(
                PreparedFrame(
                    frame_index=order,
                    frame_stem=frame_stem,
                    inference_path=str(inference_path),
                    original_path=str(original_path),
                    original_width=original_width,
                    original_height=original_height,
                    inference_width=inference_width,
                    inference_height=inference_height,
                )
            )
        return prepared

    relative_video_path = str(_field(video_info, "relative_path"))
    source_video = data_root / relative_video_path
    if not source_video.exists():
        raise FileNotFoundError(f"Video file does not exist: {source_video}")
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {source_video}")
    order = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        original_height, original_width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(rgb)
        inference_width, inference_height = _resize_size(original_width, original_height, resize_long_side)
        inference_image = original_image
        if (inference_width, inference_height) != original_image.size:
            inference_image = original_image.resize((inference_width, inference_height), Image.Resampling.BILINEAR)
        inference_path = inference_dir / f"{order:05d}.jpg"
        original_path = original_dir / f"{order:05d}.jpg"
        inference_image.save(inference_path, quality=95)
        original_image.save(original_path, quality=95)
        frame_stem = output_frame_stems[order] if order < len(output_frame_stems) else f"{order:05d}"
        prepared.append(
            PreparedFrame(
                frame_index=order,
                frame_stem=frame_stem,
                inference_path=str(inference_path),
                original_path=str(original_path),
                original_width=original_width,
                original_height=original_height,
                inference_width=inference_width,
                inference_height=inference_height,
            )
        )
        order += 1
    cap.release()
    if not prepared:
        raise RuntimeError(f"No frames decoded from video file: {source_video}")
    return prepared


def _load_mask_prompt(init_prompts: Iterable[Any], data_root: Path) -> np.ndarray:
    """Load the first mask prompt from detected prompt metadata."""

    for prompt in init_prompts:
        if str(_field(prompt, "prompt_type", "")).lower() != "mask":
            continue
        mask_path = data_root / str(_field(prompt, "relative_path"))
        if not mask_path.exists():
            raise FileNotFoundError(f"Initial mask prompt file does not exist: {mask_path}")
        return np.asarray(Image.open(mask_path))
    raise RuntimeError("SAM2 baseline requires an initial first-frame mask prompt; none was detected.")


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an indexed/binary mask using nearest-neighbor interpolation."""

    image = Image.fromarray(mask.astype(np.uint8))
    if image.size == (width, height):
        return np.asarray(image)
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))


def _object_masks_from_initial_mask(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Split a binary or indexed first-frame mask into per-object boolean masks."""

    if mask.ndim == 3:
        mask = np.any(mask[..., :3] > 0, axis=-1).astype(np.uint8)
    unique_values = sorted(int(value) for value in np.unique(mask).tolist())
    positives = [value for value in unique_values if value > 0]
    if not positives:
        raise RuntimeError("Initial mask prompt contains no foreground pixels.")
    if set(unique_values).issubset({0, 1, 255}) and len(positives) <= 2:
        return {1: mask > 0}
    return {object_id: mask == object_id for object_id in positives if object_id != 255}


def _bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    """Compute inclusive ``[xmin, ymin, xmax, ymax]`` box for a boolean mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Cannot compute a bounding box for an empty mask.")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _pad_box(box: np.ndarray, width: int, height: int, rho: float) -> np.ndarray:
    """Pad a box by a fraction of its width and height and clip to image bounds."""

    xmin, ymin, xmax, ymax = box.astype(np.float32)
    box_w = max(1.0, xmax - xmin + 1.0)
    box_h = max(1.0, ymax - ymin + 1.0)
    padded = np.array(
        [xmin - rho * box_w, ymin - rho * box_h, xmax + rho * box_w, ymax + rho * box_h],
        dtype=np.float32,
    )
    padded[[0, 2]] = np.clip(padded[[0, 2]], 0, width - 1)
    padded[[1, 3]] = np.clip(padded[[1, 3]], 0, height - 1)
    return padded


def _positive_points(mask: np.ndarray, count: int) -> np.ndarray:
    """Select interior positive points by distance transform."""

    count = max(0, int(count))
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    binary = mask.astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    values = distance[ys, xs]
    order = np.argsort(-values)
    selected: list[tuple[float, float]] = []
    min_spacing = max(1.0, math.sqrt(float(binary.sum())) / max(count, 1) / 2.0)
    for idx in order:
        point = (float(xs[idx]), float(ys[idx]))
        if all((point[0] - prev[0]) ** 2 + (point[1] - prev[1]) ** 2 >= min_spacing**2 for prev in selected):
            selected.append(point)
        if len(selected) >= count:
            break
    if not selected:
        selected.append((float(xs[order[0]]), float(ys[order[0]])))
    while len(selected) < count:
        selected.append(selected[-1])
    return np.asarray(selected[:count], dtype=np.float32)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask with an elliptical kernel."""

    radius = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _negative_points(mask: np.ndarray, count: int, r1: int, r2: int) -> np.ndarray:
    """Sample negative points from the ring between two dilations."""

    count = max(0, int(count))
    if count == 0:
        return np.zeros((0, 2), dtype=np.float32)
    outer = _dilate(mask, max(r1, r2))
    inner = _dilate(mask, min(r1, r2))
    ring = outer & ~inner
    ys, xs = np.where(ring)
    if len(xs) == 0:
        ys, xs = np.where(~mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    center_y, center_x = np.argwhere(mask).mean(axis=0)
    angles = np.arctan2(ys - center_y, xs - center_x)
    order = np.argsort(angles)
    chosen_indices = np.linspace(0, len(order) - 1, num=min(count, len(order)), dtype=int)
    points = [(float(xs[order[idx]]), float(ys[order[idx]])) for idx in chosen_indices]
    while len(points) < count:
        points.append(points[-1])
    return np.asarray(points[:count], dtype=np.float32)


def build_object_prompts(mask: np.ndarray, config: dict[str, Any]) -> list[ObjectPrompt]:
    """Build per-object SAM2 prompts from the first-frame mask."""

    rho = float(config.get("box_padding_ratio", 0.08))
    num_positive = int(config.get("num_positive", 5))
    num_negative = int(config.get("num_negative", 8))
    r1 = int(config.get("negative_r1", 8))
    r2 = int(config.get("negative_r2", 25))
    height, width = mask.shape[:2]
    object_masks = _object_masks_from_initial_mask(mask)
    prompts: list[ObjectPrompt] = []
    for object_id, object_mask in sorted(object_masks.items()):
        box = _bbox_from_mask(object_mask)
        padded_box = _pad_box(box, width, height, rho)
        pos = _positive_points(object_mask, num_positive)
        neg = _negative_points(object_mask, num_negative, r1, r2)
        points = np.concatenate([pos, neg], axis=0).astype(np.float32)
        labels = np.concatenate(
            [np.ones((len(pos),), dtype=np.int32), np.zeros((len(neg),), dtype=np.int32)],
            axis=0,
        )
        prompts.append(
            ObjectPrompt(
                object_id=int(object_id),
                mask=object_mask.astype(bool),
                box=padded_box.astype(np.float32),
                points=points,
                labels=labels,
                original_box=[int(value) for value in box.tolist()],
            )
        )
    return prompts


def _autocast_dtype(device: str, warnings: list[str]) -> Any:
    """Choose bfloat16 when supported, otherwise float16."""

    torch = _import_torch()
    if not device.startswith("cuda"):
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    warnings.append("CUDA bfloat16 is not supported; falling back to float16 autocast.")
    return torch.float16


def _add_prompt_to_predictor(
    predictor: Any,
    inference_state: Any,
    prompt: ObjectPrompt,
    prompt_mode: str,
    warnings: list[str],
) -> None:
    """Add one object prompt to the SAM2 predictor."""

    use_mask = prompt_mode in {"mask", "mask_box_points"} and hasattr(predictor, "add_new_mask")
    if use_mask:
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=prompt.object_id,
            mask=prompt.mask,
        )
        if prompt_mode == "mask_box_points":
            warnings.append(
                "SAM2 video API stores either mask or point/box prompts per frame; "
                "mask_box_points used add_new_mask as the primary prompt."
            )
        return
    if prompt_mode == "mask":
        warnings.append("predictor.add_new_mask is unavailable; falling back from mask to box_points.")
        prompt_mode = "box_points"

    if prompt_mode == "box":
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=prompt.object_id,
            box=prompt.box,
        )
        return
    if prompt_mode == "points":
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=prompt.object_id,
            points=prompt.points,
            labels=prompt.labels,
        )
        return
    if prompt_mode in {"box_points", "mask_box_points"}:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=prompt.object_id,
            points=prompt.points,
            labels=prompt.labels,
            box=prompt.box,
        )
        return
    raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Return a bounding box for a boolean mask, or None for an empty mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _bbox_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    """Compute IoU for two inclusive boxes."""

    if box_a is None or box_b is None:
        return 0.0
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x1 - x0 + 1.0)
    inter_h = max(0.0, y1 - y0 + 1.0)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0] + 1.0)) * max(0.0, float(box_a[3] - box_a[1] + 1.0))
    area_b = max(0.0, float(box_b[2] - box_b[0] + 1.0)) * max(0.0, float(box_b[3] - box_b[1] + 1.0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _compose_binary_masks_with_iou(
    binary_masks: np.ndarray,
    object_ids: list[int],
    previous_indexed_mask: np.ndarray | None,
) -> np.ndarray:
    """Compose binary masks by previous-frame bbox IoU when scores are unavailable."""

    output = np.zeros(binary_masks.shape[-2:], dtype=np.uint8)
    if previous_indexed_mask is None:
        for index, object_id in enumerate(object_ids):
            output[(binary_masks[index] > 0) & (output == 0)] = np.uint8(min(max(object_id, 0), 255))
        return output

    score_planes = np.full(binary_masks.shape, -np.inf, dtype=np.float32)
    for index, object_id in enumerate(object_ids):
        current_box = _mask_bbox(binary_masks[index] > 0)
        previous_box = _mask_bbox(previous_indexed_mask == object_id)
        score = _bbox_iou(current_box, previous_box)
        score_planes[index][binary_masks[index] > 0] = score
    winners = np.argmax(score_planes, axis=0)
    max_scores = np.max(score_planes, axis=0)
    foreground = np.isfinite(max_scores)
    for index, object_id in enumerate(object_ids):
        output[foreground & (winners == index)] = np.uint8(min(max(object_id, 0), 255))
    return output


def _compose_indexed_mask(
    mask_logits: Any,
    object_ids: Iterable[int],
    previous_indexed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compose per-object masks into one indexed PNG mask."""

    if hasattr(mask_logits, "detach"):
        logits = mask_logits.detach().float().cpu().numpy()
    else:
        logits = np.asarray(mask_logits, dtype=np.float32)
    if logits.ndim == 4:
        logits = logits[:, 0]
    elif logits.ndim == 2:
        logits = logits[None]
    object_id_list = [int(obj_id) for obj_id in object_ids]
    if logits.shape[0] != len(object_id_list):
        object_id_list = object_id_list[: logits.shape[0]]
    if not object_id_list:
        return np.zeros(logits.shape[-2:], dtype=np.uint8)
    unique_values = np.unique(logits)
    if unique_values.size <= 3 and set(float(value) for value in unique_values.tolist()).issubset({0.0, 1.0}):
        return _compose_binary_masks_with_iou(logits > 0, object_id_list, previous_indexed_mask)
    active_scores = np.where(logits > 0, logits, -np.inf)
    max_scores = np.max(active_scores, axis=0)
    winners = np.argmax(active_scores, axis=0)
    output = np.zeros(logits.shape[-2:], dtype=np.uint8)
    foreground = np.isfinite(max_scores)
    for index, object_id in enumerate(object_id_list):
        output[foreground & (winners == index)] = np.uint8(min(max(object_id, 0), 255))
    return output


def _save_overlay(frame_path: Path, mask: np.ndarray, output_path: Path) -> None:
    """Save a simple RGB overlay for debugging."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(frame_path).convert("RGB")
    if mask.shape[:2] != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
    arr = np.asarray(image).copy()
    palette = [
        ImageColor.getrgb("#ff4040"),
        ImageColor.getrgb("#34c759"),
        ImageColor.getrgb("#0a84ff"),
        ImageColor.getrgb("#ffcc00"),
        ImageColor.getrgb("#bf5af2"),
        ImageColor.getrgb("#ff9f0a"),
    ]
    for index, object_id in enumerate(sorted(int(value) for value in np.unique(mask) if value > 0)):
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        region = mask == object_id
        arr[region] = (0.55 * arr[region] + 0.45 * color).astype(np.uint8)
    Image.fromarray(arr).save(output_path, quality=90)


def _save_indexed_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save an indexed mask as an 8-bit PNG."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(output_path)


def _should_save_overlay(frame_idx: int, config: dict[str, Any], warnings: list[str]) -> bool:
    """Return whether to write a debug overlay for one frame."""

    mode = str(config.get("save_overlays", "sample")).lower()
    if mode not in {"none", "sample", "all"}:
        warnings.append(f"Unknown save_overlays={mode!r}; using sample overlays.")
        mode = "sample"
    if mode == "none":
        return False
    if mode == "all":
        return True
    stride = max(1, int(config.get("overlay_stride", 12) or 12))
    return int(frame_idx) % stride == 0


def run_sam2_on_video(
    video_info: Any,
    init_prompts: Iterable[Any],
    output_dir: str | Path,
    config: dict[str, Any],
) -> Sam2VideoResult:
    """Run SAM2 on one video and write indexed masks and debug overlays."""

    torch = _import_torch()
    video_id = _video_id(video_info)
    output_root = Path(output_dir).expanduser().resolve()
    masks_dir = output_root / "masks" / video_id
    overlays_dir = output_root / "overlays" / video_id
    raw_logits_dir = output_root / "raw_logits" / video_id
    warnings: list[str] = []
    inference_state: Any | None = None

    try:
        predictor = config["predictor"]
        device = str(config.get("device", "cuda"))
        prompt_mode = str(config.get("prompt_mode", "mask_box_points"))
        save_raw_logits = bool(config.get("save_raw_logits", False))
        prepared_frames = _prepare_frame_dir(video_info, config)
        first = prepared_frames[0]
        data_root = Path(config["data_root"]).expanduser().resolve()
        init_mask = _load_mask_prompt(init_prompts, data_root)
        init_mask = _resize_mask(init_mask, first.inference_width, first.inference_height)
        object_prompts = build_object_prompts(init_mask, config)
        autocast_dtype = _autocast_dtype(device, warnings)
        object_ids = [prompt.object_id for prompt in object_prompts]
        frame_dir = Path(prepared_frames[0].inference_path).parent
        mask_paths: list[str] = []
        overlay_paths: list[str] = []
        raw_logit_paths: list[str] = []

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_dtype) if device.startswith("cuda") else _nullcontext():
                inference_state = predictor.init_state(video_path=str(frame_dir))
                for prompt in object_prompts:
                    _add_prompt_to_predictor(predictor, inference_state, prompt, prompt_mode, warnings)
                previous_indexed_mask: np.ndarray | None = None
                for frame_idx, out_obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
                    if frame_idx < 0 or frame_idx >= len(prepared_frames):
                        warnings.append(f"SAM2 returned out-of-range frame_idx={frame_idx}; skipped.")
                        continue
                    frame = prepared_frames[int(frame_idx)]
                    indexed_mask = _compose_indexed_mask(mask_logits, out_obj_ids, previous_indexed_mask)
                    if indexed_mask.shape[:2] != (frame.original_height, frame.original_width):
                        indexed_mask = np.asarray(
                            Image.fromarray(indexed_mask).resize(
                                (frame.original_width, frame.original_height),
                                Image.Resampling.NEAREST,
                            )
                        )
                    mask_path = masks_dir / f"{frame.frame_stem}.png"
                    overlay_path = overlays_dir / f"{frame.frame_stem}.jpg"
                    _save_indexed_mask(indexed_mask, mask_path)
                    mask_paths.append(str(mask_path))
                    if _should_save_overlay(int(frame_idx), config, warnings):
                        _save_overlay(Path(frame.original_path), indexed_mask, overlay_path)
                        overlay_paths.append(str(overlay_path))
                    if save_raw_logits:
                        raw_logits_dir.mkdir(parents=True, exist_ok=True)
                        raw_path = raw_logits_dir / f"{frame.frame_stem}.npz"
                        logits_np = mask_logits.detach().float().cpu().numpy() if hasattr(mask_logits, "detach") else np.asarray(mask_logits)
                        np.savez_compressed(raw_path, logits=logits_np, object_ids=np.asarray([int(v) for v in out_obj_ids]))
                        raw_logit_paths.append(str(raw_path))
                    previous_indexed_mask = indexed_mask

        return Sam2VideoResult(
            video_id=video_id,
            status="done",
            frame_count=len(prepared_frames),
            object_ids=object_ids,
            mask_paths=mask_paths,
            overlay_paths=overlay_paths,
            raw_logit_paths=raw_logit_paths,
            warnings=warnings,
        )
    except Exception as exc:
        return Sam2VideoResult(
            video_id=video_id,
            status="failed",
            frame_count=int(_field(video_info, "frame_count", 0) or 0),
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if inference_state is not None:
            del inference_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _nullcontext:
    """Minimal null context manager to avoid importing contextlib in hot paths."""

    def __enter__(self) -> None:
        """Enter a no-op context."""

        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Exit a no-op context."""

        return False

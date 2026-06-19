"""Optional Cutie adapter for mask-conditioned VOS candidates."""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from src.trackers.sam2_tracker import (
    _load_mask_prompt,
    _prepare_frame_dir,
    _resize_mask,
    _save_indexed_mask,
    _save_overlay,
    _should_save_overlay,
)


CUTIE_REPO_URL = "https://github.com/hkchengrex/Cutie.git"


@dataclass(slots=True)
class CutieAvailability:
    """Structured Cutie runtime availability."""

    available: bool
    reason: str
    repo_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CutieVideoResult:
    """Per-video Cutie candidate result."""

    video_id: str
    status: str
    frame_count: int
    object_ids: list[int] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    overlay_paths: list[str] = field(default_factory=list)
    diagnostics_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    first_frame_exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_cutie_available() -> CutieAvailability:
    """Return whether the Cutie package can be imported."""

    spec = importlib.util.find_spec("cutie")
    if spec is None:
        return CutieAvailability(False, "cutie package is not importable")
    origin = str(spec.origin) if spec.origin else None
    repo_path = None
    if origin:
        path = Path(origin).resolve()
        repo_path = str(path.parents[1]) if len(path.parents) > 1 else str(path.parent)
    return CutieAvailability(True, "available", repo_path=repo_path)


def install_or_check_cutie(repo_dir: str | Path | None = None, install: bool = False) -> CutieAvailability:
    """Make Cutie importable, optionally cloning and installing the official repo."""

    status = check_cutie_available()
    if status.available:
        return status
    if not install:
        return status
    if repo_dir is None:
        return CutieAvailability(False, "cutie is missing and no repo_dir was supplied for installation")

    repo_path = Path(repo_dir).expanduser().resolve()
    try:
        if repo_path.exists() and not (repo_path / "cutie").exists():
            shutil.rmtree(repo_path)
        if not repo_path.exists():
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["git", "clone", "--depth", "1", CUTIE_REPO_URL, str(repo_path)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_path)])
        importlib.invalidate_caches()
    except Exception as exc:
        return CutieAvailability(False, "failed to install cutie", repo_path=str(repo_path), error=f"{type(exc).__name__}: {exc}")
    return check_cutie_available()


def cutie_object_ids_from_indexed(indexed: np.ndarray) -> list[int]:
    """Return positive object ids from an indexed first-frame mask."""

    ids = sorted(int(value) for value in np.unique(indexed).tolist() if int(value) > 0)
    illegal = [value for value in ids if value > 255]
    if illegal:
        raise ValueError(f"Cutie candidate expects 8-bit indexed object ids; found {illegal[:10]}")
    return ids


def sanitize_cutie_prediction(indexed: np.ndarray, object_ids: Iterable[int]) -> tuple[np.ndarray, list[int]]:
    """Clear unexpected object ids from a Cutie prediction."""

    allowed = {0, *[int(object_id) for object_id in object_ids]}
    output = np.asarray(indexed).astype(np.uint8, copy=True)
    unexpected = sorted(int(value) for value in np.unique(output).tolist() if int(value) not in allowed)
    if unexpected:
        output[~np.isin(output, list(allowed))] = 0
    return output, unexpected


def build_cutie_model(device: str = "cuda") -> Any:
    """Build the official default Cutie model."""

    try:
        get_default = importlib.import_module("cutie.utils.get_default_model").get_default_model
    except Exception as exc:
        raise RuntimeError("Could not import cutie.utils.get_default_model.get_default_model") from exc
    model = get_default()
    to_device = getattr(model, "to", None)
    if callable(to_device):
        model = to_device(device)
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
    return model


def _build_processor(model: Any, max_internal_size: int) -> Any:
    inference_core = importlib.import_module("cutie.inference.inference_core").InferenceCore
    processor = inference_core(model, cfg=getattr(model, "cfg", None))
    processor.max_internal_size = int(max_internal_size)
    return processor


def _image_to_tensor(image_path: str | Path, device: str) -> Any:
    try:
        import torch
        from torchvision.transforms.functional import to_tensor

        image = Image.open(image_path).convert("RGB")
        return to_tensor(image).to(device=device).float()
    except Exception:
        import torch

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        return tensor.to(device=device).float()


def _tensor_to_indexed(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[0]
    return array.astype(np.uint8)


def _autocast_context(device: str) -> Any:
    if not str(device).startswith("cuda"):
        return nullcontext()
    try:
        import torch

        return torch.cuda.amp.autocast()
    except Exception:
        return nullcontext()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(path)


def run_cutie_on_video(video_info: Any, prompts: Iterable[Any], output_root: str | Path, config: dict[str, Any]) -> CutieVideoResult:
    """Run Cutie on one video and write indexed candidate masks."""

    import torch

    video_id = str(video_info.video_id)
    output_root = Path(output_root)
    masks_dir = output_root / "masks" / video_id
    overlays_dir = output_root / "overlays" / video_id
    logs_dir = output_root / "logs" / "cutie_frames" / video_id
    warnings: list[str] = []
    frame_rows: list[dict[str, Any]] = []
    mask_paths: list[str] = []
    overlay_paths: list[str] = []

    try:
        frames = _prepare_frame_dir(video_info, config)
        if not frames:
            raise RuntimeError(f"{video_id}: no frames found")
        target_count = int(config.get("max_frames") or 0)
        if target_count > 0:
            frames = frames[:target_count]
        output_frame_stems = [str(stem) for stem in config.get("output_frame_stems", [])]
        first = frames[0]
        initial = _load_mask_prompt(prompts, Path(config["data_root"]).expanduser().resolve())
        if initial.ndim == 3:
            initial = np.any(initial[..., :3] > 0, axis=-1).astype(np.uint8)
        initial = _resize_mask(initial, first.original_width, first.original_height).astype(np.uint8)
        object_ids = cutie_object_ids_from_indexed(initial)
        if not object_ids:
            raise RuntimeError(f"{video_id}: first-frame prompt has no object ids")

        device = str(config.get("device", "cuda"))
        model = config.get("cutie_model")
        if model is None:
            model = build_cutie_model(device=device)
        processor = _build_processor(model, int(config.get("max_internal_size", 720)))
        mask_tensor = torch.from_numpy(initial).to(device=device)

        with torch.inference_mode(), _autocast_context(device):
            for frame_idx, frame in enumerate(frames):
                image_tensor = _image_to_tensor(frame.original_path, device=device)
                frame_stem = output_frame_stems[frame_idx] if frame_idx < len(output_frame_stems) else frame.frame_stem
                if frame_idx == 0:
                    output_prob = processor.step(image_tensor, mask_tensor, objects=object_ids)
                    indexed = initial.copy()
                else:
                    output_prob = processor.step(image_tensor)
                    indexed = _tensor_to_indexed(processor.output_prob_to_mask(output_prob))
                    if indexed.shape[:2] != (frame.original_height, frame.original_width):
                        indexed = np.asarray(
                            Image.fromarray(indexed).resize((frame.original_width, frame.original_height), Image.Resampling.NEAREST)
                        ).astype(np.uint8)
                    indexed, unexpected = sanitize_cutie_prediction(indexed, object_ids)
                    if unexpected:
                        warnings.append(f"{video_id}:{frame_stem}: cleared unexpected Cutie object ids {unexpected[:10]}")

                output_path = masks_dir / f"{frame_stem}.png"
                _save_indexed_mask(indexed, output_path)
                mask_paths.append(str(output_path))
                present = sorted(int(value) for value in np.unique(indexed).tolist() if int(value) > 0)
                missing = [int(object_id) for object_id in object_ids if int(object_id) not in present]
                frame_rows.append(
                    {
                        "video_id": video_id,
                        "frame_stem": frame_stem,
                        "frame_index": int(frame_idx),
                        "present_object_ids": present,
                        "missing_object_ids": missing,
                        "foreground_pixels": int((indexed > 0).sum()),
                    }
                )
                if _should_save_overlay(int(frame_idx), config, warnings):
                    overlay_path = overlays_dir / f"{frame_stem}.jpg"
                    _save_overlay(Path(frame.original_path), indexed, overlay_path)
                    overlay_paths.append(str(overlay_path))

        diagnostics_path = logs_dir / "frames.json"
        _atomic_json(diagnostics_path, {"video_id": video_id, "object_ids": object_ids, "frames": frame_rows})
        return CutieVideoResult(
            video_id=video_id,
            status="done",
            frame_count=len(frames),
            object_ids=object_ids,
            mask_paths=mask_paths,
            overlay_paths=overlay_paths,
            diagnostics_path=str(diagnostics_path),
            warnings=warnings,
            first_frame_exact=True,
        )
    except Exception as exc:
        return CutieVideoResult(
            video_id=video_id,
            status="failed",
            frame_count=len(mask_paths),
            mask_paths=mask_paths,
            overlay_paths=overlay_paths,
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}",
        )

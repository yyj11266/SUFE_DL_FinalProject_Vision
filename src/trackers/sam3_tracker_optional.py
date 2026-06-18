"""Strict SAM 3.1 Object Multiplex adapter for semi-supervised VOS.

The submission path uses the official full Object Multiplex video predictor.
The older low-level tracker path is kept only as an explicit debug mode because
it can produce valid files while silently losing objects after the first frames.
Mask prompts are never converted to boxes or points, and setup failures never
fall back to SAM2. This keeps the SAM 3.1 experiment interpretable.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from src.trackers.sam2_tracker import (
    _compose_indexed_mask,
    _load_mask_prompt,
    _object_masks_from_initial_mask,
    _prepare_frame_dir,
    _resize_mask,
    _save_indexed_mask,
    _save_overlay,
)


SAM3_REPO_URL = "https://github.com/facebookresearch/sam3.git"
SAM31_HF_REPO = "research21/sam3.1"
SAM31_CHECKPOINT_NAME = "sam3.1_multiplex.pt"
SAM3_RUN_MODE_FULL = "full_predictor_mask"
SAM3_RUN_MODE_LOW_LEVEL = "low_level_debug"
SAM3_RUN_MODES = (SAM3_RUN_MODE_FULL, SAM3_RUN_MODE_LOW_LEVEL)
MIN_PYTHON = (3, 12)
MIN_TORCH = (2, 7)
MIN_CUDA = (12, 6)
SAM3_AVAILABLE = False


def _should_save_overlay(frame_idx: int, config: Mapping[str, Any], warnings: list[str]) -> bool:
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


@dataclass(slots=True)
class Sam3Availability:
    """Structured SAM 3.1 runtime preflight result."""

    available: bool
    reason: str
    warnings: list[str] = field(default_factory=list)
    sam3_importable: bool = False
    video_builder_available: bool = False
    full_predictor_builder_available: bool = False
    low_level_video_builder_available: bool = False
    image_builder_available: bool = False
    torch_available: bool = False
    cuda_available: bool = False
    checkpoint_available: bool = False
    hf_token_present: bool = False
    repo_path: str | None = None
    error: str | None = None
    python_version: str = platform.python_version()
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    native_mask_api: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Sam3TrackerBuildResult:
    """Result of building the official SAM 3.1 multiplex model."""

    available: bool
    predictor: Any | None = None
    status: Sam3Availability | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    build_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status.to_dict() if self.status else None,
            "warnings": list(self.warnings),
            "error": self.error,
            "predictor_built": self.predictor is not None,
            "build_config": dict(self.build_config),
        }


@dataclass(slots=True)
class Sam3VideoResult:
    """Per-video output metadata for native SAM 3.1 inference."""

    video_id: str
    status: str
    frame_count: int
    object_ids: list[int] = field(default_factory=list)
    mask_paths: list[str] = field(default_factory=list)
    overlay_paths: list[str] = field(default_factory=list)
    raw_logit_paths: list[str] = field(default_factory=list)
    native_scores_path: str | None = None
    native_state_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    fallback_used: bool = False
    sam3_available: bool = True
    first_frame_exact: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    parts: list[int] = []
    for token in value.split("+")[0].split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _hf_token_present() -> bool:
    if any(os.environ.get(name) for name in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _repo_path_from_spec(spec: Any) -> str | None:
    origin = getattr(spec, "origin", None)
    if not origin:
        return None
    path = Path(origin).resolve()
    return str(path.parents[1]) if len(path.parents) > 1 else str(path.parent)


def check_sam3_available(
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    strict_runtime: bool = True,
) -> Sam3Availability:
    """Validate the official SAM 3.1 package, model builder, auth, and GPU."""

    global SAM3_AVAILABLE
    warnings: list[str] = []
    errors: list[str] = []
    spec = importlib.util.find_spec("sam3")
    if spec is None:
        SAM3_AVAILABLE = False
        return Sam3Availability(False, "sam3 package is not importable")

    python_ok = sys.version_info[:2] >= MIN_PYTHON
    if not python_ok:
        errors.append(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; found {platform.python_version()}")

    torch_available = False
    cuda_available = False
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    torch_ok = False
    cuda_ok = not str(device).startswith("cuda")
    bf16_ok = not str(device).startswith("cuda")
    try:
        import torch

        torch_available = True
        torch_version = str(torch.__version__)
        torch_ok = _version_tuple(torch_version) >= MIN_TORCH
        if not torch_ok:
            errors.append(f"PyTorch {MIN_TORCH[0]}.{MIN_TORCH[1]}+ is required; found {torch_version}")
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(torch.version.cuda) if torch.version.cuda else None
        if str(device).startswith("cuda"):
            if not cuda_available:
                errors.append("CUDA was requested but torch.cuda.is_available() is False")
            else:
                index = torch.device(device).index or 0
                gpu_name = torch.cuda.get_device_name(index)
                major, minor = torch.cuda.get_device_capability(index)
                compute_capability = f"{major}.{minor}"
                cuda_ok = _version_tuple(cuda_version) >= MIN_CUDA
                bf16_ok = major >= 8 and bool(torch.cuda.is_bf16_supported())
                if not cuda_ok:
                    errors.append(f"CUDA runtime {MIN_CUDA[0]}.{MIN_CUDA[1]}+ is required; found {cuda_version}")
                if not bf16_ok:
                    errors.append(f"A BF16-capable Ampere-or-newer GPU is required; found {gpu_name} (sm_{major}{minor})")
    except Exception as exc:
        errors.append(f"PyTorch preflight failed: {type(exc).__name__}: {exc}")

    video_builder = False
    low_level_video_builder = False
    image_builder = False
    native_mask_api = False
    import_error: str | None = None
    try:
        builder_mod = importlib.import_module("sam3.model_builder")
        video_builder = hasattr(builder_mod, "build_sam3_multiplex_video_predictor")
        low_level_video_builder = hasattr(builder_mod, "build_sam3_multiplex_video_model")
        image_builder = hasattr(builder_mod, "build_sam3_image_model")
        if not video_builder:
            errors.append("sam3.model_builder.build_sam3_multiplex_video_predictor is missing")
    except Exception as exc:
        import_error = f"{type(exc).__name__}: {exc}"
        errors.append(f"Could not import sam3.model_builder: {import_error}")

    checkpoint_available = False
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
        checkpoint_available = checkpoint.is_file() and checkpoint.stat().st_size > 0
        if not checkpoint_available:
            errors.append(f"SAM 3.1 checkpoint is missing or empty: {checkpoint}")
    hf_token_present = _hf_token_present()
    if not checkpoint_path and not hf_token_present:
        warnings.append(f"No local checkpoint and no Hugging Face token; public repo {SAM31_HF_REPO} may still download.")

    if not strict_runtime:
        runtime_errors = [
            message
            for message in errors
            if message.startswith(("Python ", "PyTorch ", "CUDA runtime ", "A BF16-capable"))
        ]
        for message in runtime_errors:
            errors.remove(message)
            warnings.append(f"Unsupported runtime allowed: {message}")

    requested_device_available = not str(device).startswith("cuda") or cuda_available
    version_gate = (python_ok and torch_ok and cuda_ok) if strict_runtime else True
    available = bool(
        video_builder
        and torch_available
        and requested_device_available
        and version_gate
        and bf16_ok
        and not errors
    )
    reason = "available" if available else "; ".join(errors)
    SAM3_AVAILABLE = available
    return Sam3Availability(
        available=available,
        reason=reason,
        warnings=warnings,
        sam3_importable=True,
        video_builder_available=video_builder,
        full_predictor_builder_available=video_builder,
        low_level_video_builder_available=low_level_video_builder,
        image_builder_available=image_builder,
        torch_available=torch_available,
        cuda_available=cuda_available,
        checkpoint_available=checkpoint_available,
        hf_token_present=hf_token_present,
        repo_path=_repo_path_from_spec(spec),
        error=import_error,
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        compute_capability=compute_capability,
        native_mask_api=native_mask_api,
    )


def install_sam3_if_requested(repo_dir: str | Path, requested: bool | None = None) -> Sam3Availability:
    """Install the official repository only when explicitly requested."""

    install_requested = _env_truthy("SUFE_INSTALL_SAM3") if requested is None else bool(requested)
    if not install_requested:
        return check_sam3_available()

    repo_path = Path(repo_dir).expanduser().resolve()
    try:
        if not (repo_path / "sam3" / "__init__.py").exists():
            repo_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["git", "clone", "--depth", "1", SAM3_REPO_URL, str(repo_path)])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_path)])
        repo_text = str(repo_path)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        importlib.invalidate_caches()
    except Exception as exc:
        return Sam3Availability(
            available=False,
            reason="SAM 3.1 installation failed",
            repo_path=str(repo_path),
            error=f"{type(exc).__name__}: {exc}",
        )
    status = check_sam3_available()
    status.repo_path = str(repo_path)
    return status


def _require_native_api(model: Any) -> None:
    required = ("init_state", "add_new_masks", "propagate_in_video")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise RuntimeError(
            "SAM 3.1 native mask conditioning API is incomplete; missing "
            f"{missing}. Refusing box/point/SAM2 fallback."
        )


def _full_predictor_model(predictor: Any) -> Any:
    model = getattr(predictor, "model", None)
    if model is None:
        raise RuntimeError("SAM 3.1 full predictor is missing its high-level .model object")
    return model


def _require_full_predictor_mask_api(predictor: Any) -> None:
    model = _full_predictor_model(predictor)
    required = (
        "init_state",
        "propagate_in_video",
        "_tracker_add_new_objects",
        "_cache_frame_outputs",
        "_initialize_metadata",
        "add_action_history",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise RuntimeError(
            "SAM 3.1 full predictor mask conditioning API is incomplete; missing "
            f"{missing}. Refusing low-level or point/box fallback."
        )


def _short_error(exc: BaseException, limit: int = 600) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


def _load_checkpoint_mapping(checkpoint_path: str | Path) -> Mapping[str, Any]:
    import torch

    checkpoint = Path(checkpoint_path).expanduser()
    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and "model" in loaded and isinstance(loaded["model"], Mapping):
        loaded = loaded["model"]
    if not isinstance(loaded, Mapping):
        raise RuntimeError(f"SAM 3.1 checkpoint has unsupported payload type: {type(loaded).__name__}")
    return loaded


def _remap_full_multiplex_checkpoint_for_native_tracker(model: Any, checkpoint: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert full SAM 3.1 multiplex demo checkpoints into low-level tracker keys."""

    import torch

    model_state = model.state_dict()
    remapped: dict[str, Any] = {}
    candidates = 0
    skipped_missing = 0
    skipped_shape = 0
    source_prefix_counts: dict[str, int] = {}

    def candidate_key(key: str) -> tuple[str | None, str | None]:
        if key.startswith("tracker.model."):
            return key[len("tracker.model.") :], "tracker.model"
        if key.startswith("sam2_predictor.model."):
            return key[len("sam2_predictor.model.") :], "sam2_predictor.model"
        if key.startswith("sam2_predictor."):
            return key[len("sam2_predictor.") :], "sam2_predictor"
        if key.startswith("detector.backbone.vision_backbone."):
            return "backbone.vision_backbone." + key[len("detector.backbone.vision_backbone.") :], "detector.backbone.vision_backbone"
        if key in model_state:
            return key, "direct"
        return None, None

    for source_key, value in checkpoint.items():
        target_key, source_prefix = candidate_key(str(source_key))
        if target_key is None or source_prefix is None:
            continue
        candidates += 1
        if target_key not in model_state:
            skipped_missing += 1
            continue
        expected = model_state[target_key]
        if isinstance(value, torch.Tensor) and isinstance(expected, torch.Tensor) and value.shape != expected.shape:
            skipped_shape += 1
            continue
        remapped[target_key] = value
        source_prefix_counts[source_prefix] = source_prefix_counts.get(source_prefix, 0) + 1

    expected_keys = set(model_state)
    loaded_keys = set(remapped)
    missing_keys = sorted(expected_keys - loaded_keys)
    critical_prefixes = (
        "backbone.vision_backbone.",
        "maskmem_backbone.",
        "transformer.",
        "sam_prompt_encoder.",
        "sam_mask_decoder.",
    )
    missing_critical_prefixes = [
        prefix
        for prefix in critical_prefixes
        if any(key.startswith(prefix) for key in expected_keys) and not any(key.startswith(prefix) for key in loaded_keys)
    ]
    for key in ("maskmem_tpos_enc", "interactivity_no_mem_embed", "no_obj_embed_spatial"):
        if key in expected_keys and key not in loaded_keys:
            missing_critical_prefixes.append(key)

    diagnostics = {
        "checkpoint_keys": len(checkpoint),
        "model_keys": len(model_state),
        "candidate_keys": candidates,
        "loaded_keys": len(remapped),
        "missing_keys": len(missing_keys),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "source_prefix_counts": source_prefix_counts,
        "missing_critical_prefixes": missing_critical_prefixes,
        "missing_key_examples": missing_keys[:20],
    }
    if missing_critical_prefixes:
        raise RuntimeError(
            "SAM 3.1 checkpoint remap did not load required tracker components: "
            f"{missing_critical_prefixes}. Diagnostics: {diagnostics}"
        )
    if not remapped:
        raise RuntimeError(f"SAM 3.1 checkpoint remap produced no usable tracker keys. Diagnostics: {diagnostics}")
    return remapped, diagnostics


def _build_native_tracker_from_full_checkpoint(builder_mod: Any, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    builder = getattr(builder_mod, "build_sam3_multiplex_video_model")
    build_config = dict(config)
    checkpoint_path = build_config.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("Full-checkpoint native tracker remap requires a local checkpoint")

    build_config.pop("run_mode", None)
    build_config["checkpoint_path"] = None
    build_config["load_from_HF"] = False
    build_config["strict_state_dict_loading"] = False
    model = builder(**build_config)
    checkpoint = _load_checkpoint_mapping(str(checkpoint_path))
    remapped_state, diagnostics = _remap_full_multiplex_checkpoint_for_native_tracker(model, checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state, strict=False)
    diagnostics["post_load_missing_keys"] = len(missing_keys)
    diagnostics["post_load_unexpected_keys"] = len(unexpected_keys)
    diagnostics["post_load_missing_examples"] = list(missing_keys[:20])
    diagnostics["post_load_unexpected_examples"] = list(unexpected_keys[:20])
    model.to(device=config.get("device", "cuda"))
    model.eval()
    _require_native_api(model)
    _patch_native_tracker_forward_image_for_mask_tracking(model)
    return model, diagnostics


def _patch_native_tracker_forward_image_for_mask_tracking(model: Any) -> None:
    """Avoid SAM3 multiplex low-level tracker requesting root SAM3 detector features.

    ``Sam3VideoTrackingMultiplexDemo._get_image_feature`` requests sam3, interactive,
    and propagation outputs together. The low-level ``forward_image`` clone loop only
    handles nested interactive/propagation outputs, while ``need_sam3_out=True`` adds
    root-level tensor entries. For first-frame mask tracking we only need the
    interactive and propagation branches, so force ``need_sam3_out=False``.
    """

    if getattr(model, "_sufe_forward_image_mask_tracking_patch", False):
        return
    original_forward_image = model.forward_image

    def forward_image_mask_tracking(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("need_sam3_out", False):
            kwargs = dict(kwargs)
            kwargs["need_sam3_out"] = False
        return original_forward_image(*args, **kwargs)

    object.__setattr__(model, "forward_image", forward_image_mask_tracking)
    object.__setattr__(model, "_sufe_forward_image_mask_tracking_patch", True)


def build_sam3_tracker(
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
    multiplex_count: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    compile_model: bool = False,
    strict_runtime: bool = True,
    run_mode: str = SAM3_RUN_MODE_FULL,
) -> Sam3TrackerBuildResult:
    """Build the official SAM 3.1 Object Multiplex tracking model."""

    if run_mode not in SAM3_RUN_MODES:
        return Sam3TrackerBuildResult(False, error=f"Unsupported SAM 3.1 run mode: {run_mode}")
    status = check_sam3_available(checkpoint_path, device=device, strict_runtime=strict_runtime)
    if not status.available:
        return Sam3TrackerBuildResult(False, status=status, warnings=status.warnings, error=status.reason)

    config = {
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else None,
        "multiplex_count": max(1, int(multiplex_count)),
        "use_fa3": bool(use_fa3),
        "use_rope_real": bool(use_rope_real),
        "device": device,
        "compile": bool(compile_model),
        "run_mode": run_mode,
    }
    if run_mode == SAM3_RUN_MODE_FULL:
        full_config = {
            "checkpoint_path": config["checkpoint_path"],
            "max_num_objects": max(16, int(multiplex_count)),
            "multiplex_count": config["multiplex_count"],
            "use_fa3": config["use_fa3"],
            "use_rope_real": config["use_rope_real"],
            "compile": config["compile"],
            "warm_up": False,
            "default_output_prob_thresh": 0.5,
            "async_loading_frames": False,
        }
        try:
            builder_mod = importlib.import_module("sam3.model_builder")
            builder = getattr(builder_mod, "build_sam3_multiplex_video_predictor")
            predictor = builder(**full_config)
            _require_full_predictor_mask_api(predictor)
            status.native_mask_api = True
            build_config = {**config, **full_config, "builder": "build_sam3_multiplex_video_predictor"}
            return Sam3TrackerBuildResult(
                True,
                predictor=predictor,
                status=status,
                warnings=status.warnings,
                build_config=build_config,
            )
        except Exception as exc:
            failed_config = dict(config)
            failed_config["builder"] = "build_sam3_multiplex_video_predictor"
            return Sam3TrackerBuildResult(
                False,
                status=status,
                warnings=status.warnings,
                error=_short_error(exc),
                build_config=failed_config,
            )

    low_level_config = {
        **config,
        "load_from_HF": checkpoint_path is None,
        "strict_state_dict_loading": True,
    }
    try:
        builder_mod = importlib.import_module("sam3.model_builder")
        builder = getattr(builder_mod, "build_sam3_multiplex_video_model")
        builder_config = dict(low_level_config)
        builder_config.pop("run_mode", None)
        model = builder(**builder_config)
        _require_native_api(model)
        _patch_native_tracker_forward_image_for_mask_tracking(model)
        status.native_mask_api = True
        low_level_config["builder"] = "build_sam3_multiplex_video_model/low_level_debug"
        return Sam3TrackerBuildResult(
            True,
            predictor=model,
            status=status,
            warnings=status.warnings,
            build_config=low_level_config,
        )
    except Exception as exc:
        direct_error = _short_error(exc)
        if checkpoint_path:
            try:
                builder_mod = importlib.import_module("sam3.model_builder")
                model, diagnostics = _build_native_tracker_from_full_checkpoint(builder_mod, low_level_config)
                status.native_mask_api = True
                fallback_config = dict(low_level_config)
                fallback_config["builder"] = "build_sam3_multiplex_video_model/full_checkpoint_remap_low_level_debug"
                fallback_config["direct_builder_error"] = direct_error
                fallback_config["checkpoint_remap"] = diagnostics
                warnings = list(status.warnings)
                warnings.append(
                    "Low-level debug mode loaded the full multiplex checkpoint with tracker/backbone key remapping. "
                    "This mode is rejected for submissions."
                )
                return Sam3TrackerBuildResult(
                    True,
                    predictor=model,
                    status=status,
                    warnings=warnings,
                    build_config=fallback_config,
                )
            except Exception as remap_exc:
                remap_error = _short_error(remap_exc)
                low_level_config = dict(low_level_config)
                low_level_config["direct_builder_error"] = direct_error
                low_level_config["full_checkpoint_remap_error"] = remap_error
        return Sam3TrackerBuildResult(
            False,
            status=status,
            warnings=status.warnings,
            error=low_level_config.get("full_checkpoint_remap_error") or direct_error,
            build_config=low_level_config,
        )


def _field(item: Any, name: str, default: Any = None) -> Any:
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, Mapping):
        return item.get(name, default)
    return default


def _video_id(video_info: Any) -> str:
    return str(_field(video_info, "video_id", ""))


def _to_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _sigmoid(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _probabilities_to_logits(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    probabilities = np.asarray(values, dtype=np.float32).reshape(-1)
    if probabilities.size == 0:
        return probabilities
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _call_supported(function: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(function)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def _initialize_native_state(model: Any, frame_dir: Path, offload_video: bool, offload_state: bool) -> Any:
    """Initialize current SAM 3.1 states, including its required image tensor cache."""

    signature = inspect.signature(model.init_state)
    if "video_path" in signature.parameters or "resource_path" in signature.parameters:
        return _call_supported(
            model.init_state,
            video_path=str(frame_dir),
            resource_path=str(frame_dir),
            offload_video_to_cpu=offload_video,
            offload_state_to_cpu=offload_state,
            async_loading_frames=False,
            use_cv2=False,
        )

    required = {"video_height", "video_width", "num_frames"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(f"Unsupported SAM 3.1 init_state signature: {signature}")

    io_utils = importlib.import_module("sam3.model.io_utils")
    loader = getattr(io_utils, "load_video_frames", None)
    if not callable(loader):
        raise RuntimeError("sam3.model.io_utils.load_video_frames is missing")
    loaded = _call_supported(
        loader,
        video_path=str(frame_dir),
        image_size=int(getattr(model, "image_size", 1008)),
        offload_video_to_cpu=offload_video,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
    )
    if not isinstance(loaded, Sequence) or len(loaded) < 3:
        raise RuntimeError("Official SAM 3.1 frame loader returned an unexpected payload")
    images, video_height, video_width = loaded[:3]
    state = _call_supported(
        model.init_state,
        video_height=int(video_height),
        video_width=int(video_width),
        num_frames=len(images),
        cached_features=None,
        offload_video_to_cpu=offload_video,
        offload_state_to_cpu=offload_state,
    )
    if not isinstance(state, dict):
        raise RuntimeError("SAM 3.1 init_state did not return a dictionary state")
    state["images"] = images
    return state


def _session_frame_dir(
    prepared_frames: Sequence[Any],
    target_frame_count: int,
    cache_dir: Path,
    video_id: str,
) -> Path:
    """Create an isolated numeric frame folder with exactly the requested frames."""

    session_dir = cache_dir / "sam31_sessions" / video_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for existing in session_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    for index, frame in enumerate(prepared_frames[:target_frame_count]):
        source = Path(frame.inference_path)
        target = session_dir / f"{index:05d}{source.suffix.lower()}"
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    return session_dir


def _normalize_full_predictor_outputs(frame_idx: int, outputs: Mapping[str, Any]) -> tuple[int, list[int], np.ndarray, np.ndarray | None]:
    obj_ids = outputs.get("out_obj_ids", outputs.get("obj_ids", outputs.get("object_ids")))
    masks = outputs.get("out_binary_masks", outputs.get("video_res_masks", outputs.get("mask_logits", outputs.get("masks"))))
    if obj_ids is None or masks is None:
        raise RuntimeError("SAM 3.1 full predictor output is missing out_obj_ids or out_binary_masks")
    ids = [int(value) for value in _to_numpy(obj_ids, np.int64).reshape(-1).tolist()]
    logits = _to_numpy(masks, np.float32)
    if logits.ndim == 4:
        logits = logits[:, 0]
    if logits.ndim == 2:
        logits = logits[None]
    score_array: np.ndarray | None = None
    if "out_probs" in outputs:
        score_array = _probabilities_to_logits(_to_numpy(outputs["out_probs"], np.float32))
    elif "object_score_logits" in outputs:
        score_array = _to_numpy(outputs["object_score_logits"], np.float32).reshape(-1)
    elif "presence_logits" in outputs:
        score_array = _to_numpy(outputs["presence_logits"], np.float32).reshape(-1)
    return frame_idx, ids, logits, score_array


def _normalize_propagation_item(item: Any) -> tuple[int, list[int], np.ndarray, np.ndarray | None]:
    if isinstance(item, Mapping):
        if "outputs" in item and isinstance(item["outputs"], Mapping):
            return _normalize_full_predictor_outputs(int(item.get("frame_idx", item.get("frame_index", 0))), item["outputs"])
        if "out_binary_masks" in item or "out_obj_ids" in item:
            return _normalize_full_predictor_outputs(int(item.get("frame_idx", item.get("frame_index", 0))), item)
        frame_idx = int(item.get("frame_idx", item.get("frame_index", 0)))
        obj_ids = item.get("obj_ids", item.get("object_ids"))
        masks = item.get("video_res_masks", item.get("mask_logits", item.get("masks")))
        scores = item.get("object_score_logits", item.get("presence_logits"))
    elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) == 2 and isinstance(item[1], Mapping):
            return _normalize_full_predictor_outputs(int(item[0]), item[1])
        if len(item) == 5:
            frame_idx, obj_ids, _low_res, masks, scores = item
        elif len(item) == 4:
            frame_idx, obj_ids, _low_res, masks = item
            scores = None
        elif len(item) == 3:
            frame_idx, obj_ids, masks = item
            scores = None
        else:
            raise RuntimeError(f"Unexpected SAM 3.1 propagation tuple length: {len(item)}")
        frame_idx = int(frame_idx)
    else:
        raise RuntimeError(f"Unexpected SAM 3.1 propagation item: {type(item).__name__}")

    if masks is None or obj_ids is None:
        raise RuntimeError("SAM 3.1 propagation output is missing masks or object IDs")
    ids = [int(value) for value in _to_numpy(obj_ids, np.int64).reshape(-1).tolist()]
    logits = _to_numpy(masks)
    if logits.ndim == 4:
        logits = logits[:, 0]
    if logits.ndim == 2:
        logits = logits[None]
    score_array = None if scores is None else _to_numpy(scores).reshape(-1)
    return frame_idx, ids, logits, score_array


def _frame_native_output(state: Any, frame_idx: int) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    output_dict = state.get("output_dict", {})
    if not isinstance(output_dict, Mapping):
        return {}
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        outputs = output_dict.get(storage_key, {})
        if isinstance(outputs, Mapping) and frame_idx in outputs and isinstance(outputs[frame_idx], Mapping):
            return outputs[frame_idx]
    return {}


def _reorder_objects(
    logits: np.ndarray,
    output_ids: list[int],
    expected_ids: list[int],
    allow_missing: bool = False,
) -> np.ndarray:
    if len(output_ids) != logits.shape[0]:
        raise RuntimeError(f"SAM 3.1 returned {len(output_ids)} IDs for {logits.shape[0]} masks")
    extra_ids = sorted(set(output_ids) - set(expected_ids))
    duplicate_ids = len(output_ids) != len(set(output_ids))
    missing_ids = sorted(set(expected_ids) - set(output_ids))
    if extra_ids or duplicate_ids or (missing_ids and not allow_missing):
        raise RuntimeError(f"SAM 3.1 object IDs changed: expected {expected_ids}, got {output_ids}")
    positions = {object_id: index for index, object_id in enumerate(output_ids)}
    zero_mask = np.zeros(logits.shape[-2:], dtype=logits.dtype)
    return np.stack([logits[positions[object_id]] if object_id in positions else zero_mask for object_id in expected_ids], axis=0)


def _reorder_scores(
    scores: np.ndarray | None,
    output_ids: list[int],
    expected_ids: list[int],
    allow_missing: bool = False,
) -> np.ndarray | None:
    if scores is None:
        return None
    if len(scores) != len(output_ids):
        raise RuntimeError(f"SAM 3.1 returned {len(scores)} presence scores for {len(output_ids)} objects")
    extra_ids = sorted(set(output_ids) - set(expected_ids))
    duplicate_ids = len(output_ids) != len(set(output_ids))
    missing_ids = sorted(set(expected_ids) - set(output_ids))
    if extra_ids or duplicate_ids or (missing_ids and not allow_missing):
        raise RuntimeError(f"SAM 3.1 object score IDs changed: expected {expected_ids}, got {output_ids}")
    positions = {object_id: index for index, object_id in enumerate(output_ids)}
    return np.asarray([scores[positions[object_id]] if object_id in positions else -100.0 for object_id in expected_ids], dtype=np.float32)


def _init_object_diagnostics(object_ids: list[int], frame_count: int, allow_missing_objects: bool) -> dict[str, Any]:
    return {
        "frame_count": int(frame_count),
        "allow_missing_objects": bool(allow_missing_objects),
        "missing_output_events": [],
        "per_object": {
            str(int(object_id)): {
                "total_frames": 0,
                "zero_frames": 0,
                "non_first_zero_frames": 0,
                "absent_frames": 0,
                "missing_output_frames": 0,
                "first_zero_frame": None,
                "first_absent_frame": None,
                "first_missing_output_frame": None,
                "last_zero_frame": None,
                "last_absent_frame": None,
                "last_missing_output_frame": None,
                "recovers_after_zero": False,
                "recovers_after_absent": False,
                "recovers_after_missing": False,
                "max_foreground_fraction": 0.0,
                "mean_foreground_fraction_sum": 0.0,
            }
            for object_id in object_ids
        },
    }


def _record_object_diagnostics(
    diagnostics: dict[str, Any],
    frame_idx: int,
    object_ids: list[int],
    output_ids: list[int],
    logits: np.ndarray,
    object_scores: np.ndarray | None,
) -> None:
    missing_ids = sorted(set(object_ids) - set(output_ids))
    if missing_ids:
        diagnostics.setdefault("missing_output_events", []).append(
            {"frame_index": int(frame_idx), "missing_object_ids": [int(object_id) for object_id in missing_ids]}
        )
    per_object = diagnostics.setdefault("per_object", {})
    missing_set = set(missing_ids)
    for index, object_id in enumerate(object_ids):
        item = per_object.setdefault(str(int(object_id)), {})
        item["total_frames"] = int(item.get("total_frames", 0)) + 1
        binary = logits[index] > 0
        foreground_fraction = float(binary.mean())
        item["mean_foreground_fraction_sum"] = float(item.get("mean_foreground_fraction_sum", 0.0)) + foreground_fraction
        item["max_foreground_fraction"] = max(float(item.get("max_foreground_fraction", 0.0)), foreground_fraction)
        is_zero = int(binary.sum()) == 0
        if is_zero:
            item["zero_frames"] = int(item.get("zero_frames", 0)) + 1
            if frame_idx > 0:
                item["non_first_zero_frames"] = int(item.get("non_first_zero_frames", 0)) + 1
            if item.get("first_zero_frame") is None:
                item["first_zero_frame"] = int(frame_idx)
            item["last_zero_frame"] = int(frame_idx)
        elif item.get("first_zero_frame") is not None:
            item["recovers_after_zero"] = True

        score_logit = float(object_scores[index]) if object_scores is not None and index < len(object_scores) else None
        is_absent = score_logit is not None and score_logit <= 0
        if is_absent:
            item["absent_frames"] = int(item.get("absent_frames", 0)) + 1
            if item.get("first_absent_frame") is None:
                item["first_absent_frame"] = int(frame_idx)
            item["last_absent_frame"] = int(frame_idx)
        elif item.get("first_absent_frame") is not None:
            item["recovers_after_absent"] = True

        if object_id in missing_set:
            item["missing_output_frames"] = int(item.get("missing_output_frames", 0)) + 1
            if item.get("first_missing_output_frame") is None:
                item["first_missing_output_frame"] = int(frame_idx)
            item["last_missing_output_frame"] = int(frame_idx)
        elif item.get("first_missing_output_frame") is not None:
            item["recovers_after_missing"] = True


def _finalize_object_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    per_object = diagnostics.get("per_object", {})
    total_missing_output_frames = 0
    total_non_first_zero_frames = 0
    severe_objects: list[dict[str, Any]] = []
    for object_id, item in per_object.items():
        total_frames = max(int(item.get("total_frames", 0)), 1)
        item["zero_ratio"] = float(item.get("zero_frames", 0)) / total_frames
        item["absent_ratio"] = float(item.get("absent_frames", 0)) / total_frames
        item["missing_output_ratio"] = float(item.get("missing_output_frames", 0)) / total_frames
        item["mean_foreground_fraction"] = float(item.get("mean_foreground_fraction_sum", 0.0)) / total_frames
        item.pop("mean_foreground_fraction_sum", None)
        total_missing_output_frames += int(item.get("missing_output_frames", 0))
        total_non_first_zero_frames += int(item.get("non_first_zero_frames", 0))
        if item["zero_ratio"] >= 0.95 or item["missing_output_ratio"] >= 0.25:
            severe_objects.append(
                {
                    "object_id": int(object_id),
                    "zero_ratio": item["zero_ratio"],
                    "missing_output_ratio": item["missing_output_ratio"],
                    "first_zero_frame": item.get("first_zero_frame"),
                    "first_missing_output_frame": item.get("first_missing_output_frame"),
                }
            )
    diagnostics["total_missing_output_frames"] = int(total_missing_output_frames)
    diagnostics["total_non_first_zero_frames"] = int(total_non_first_zero_frames)
    diagnostics["severe_objects"] = severe_objects
    return diagnostics


def _score_rows(
    video_id: str,
    frame_idx: int,
    object_ids: list[int],
    logits: np.ndarray,
    object_scores: np.ndarray | None,
    native_output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    iou_raw = native_output.get("iou_score", native_output.get("iou_predictions"))
    iou_scores = _to_numpy(iou_raw).reshape(-1) if iou_raw is not None else np.asarray([])
    rows: list[dict[str, Any]] = []
    for index, object_id in enumerate(object_ids):
        score_logit = float(object_scores[index]) if object_scores is not None and index < len(object_scores) else None
        predicted_iou = float(iou_scores[index]) if index < len(iou_scores) else None
        binary = logits[index] > 0
        object_logits = logits[index]
        rows.append(
            {
                "video_id": video_id,
                "frame_index": frame_idx,
                "object_id": object_id,
                "presence_logit": score_logit,
                "presence": _sigmoid(score_logit),
                "predicted_iou": predicted_iou,
                "mask_logit_min": float(np.min(object_logits)),
                "mask_logit_max": float(np.max(object_logits)),
                "mask_logit_mean": float(np.mean(object_logits)),
                "foreground_pixels": int(binary.sum()),
                "foreground_fraction": float(binary.mean()),
                "object_state": "present" if score_logit is None or score_logit > 0 else "absent",
            }
        )
    return rows


def _state_summary(state: Any, object_ids: list[int], version: dict[str, Any]) -> dict[str, Any]:
    output_dict = state.get("output_dict", {}) if isinstance(state, Mapping) else {}
    cond = output_dict.get("cond_frame_outputs", {}) if isinstance(output_dict, Mapping) else {}
    non_cond = output_dict.get("non_cond_frame_outputs", {}) if isinstance(output_dict, Mapping) else {}
    per_object = state.get("output_dict_per_obj", {}) if isinstance(state, Mapping) else {}
    summary = {
        "object_ids": object_ids,
        "first_annotation_frame": state.get("first_ann_frame_idx") if isinstance(state, Mapping) else None,
        "conditioning_frames": sorted(int(value) for value in cond),
        "propagated_frames": sorted(int(value) for value in non_cond),
        "frames_already_tracked": sorted(int(value) for value in state.get("frames_already_tracked", {})) if isinstance(state, Mapping) else [],
        "per_object_state_slots": sorted(int(value) for value in per_object) if isinstance(per_object, Mapping) else [],
        "runtime": version,
    }
    if isinstance(state, Mapping):
        cached = state.get("cached_frame_outputs", {})
        tracker_metadata = state.get("tracker_metadata", {})
        sam2_states = state.get("sam2_inference_states", [])
        feature_cache = state.get("feature_cache", {})
        if isinstance(cached, Mapping):
            summary["cached_frame_outputs"] = sorted(int(value) for value in cached)
        if isinstance(tracker_metadata, Mapping):
            ids_all = tracker_metadata.get("obj_ids_all_gpu")
            summary["tracker_metadata"] = {
                "obj_ids_all_gpu": _to_numpy(ids_all, np.int64).reshape(-1).astype(int).tolist() if ids_all is not None else [],
                "num_obj_per_gpu": _to_numpy(tracker_metadata.get("num_obj_per_gpu"), np.int64).reshape(-1).astype(int).tolist(),
                "num_buc_per_gpu": _to_numpy(tracker_metadata.get("num_buc_per_gpu"), np.int64).reshape(-1).astype(int).tolist(),
                "max_obj_id": int(tracker_metadata.get("max_obj_id", -1)),
                "score_frames": sorted(int(value) for value in tracker_metadata.get("obj_id_to_sam2_score_frame_wise", {})),
            }
        if isinstance(sam2_states, Sequence) and not isinstance(sam2_states, (str, bytes)):
            summary["sam2_inference_states"] = [
                {"obj_ids": [int(value) for value in state_item.get("obj_ids", [])]}
                for state_item in sam2_states
                if isinstance(state_item, Mapping)
            ]
        if isinstance(feature_cache, Mapping):
            summary["feature_cache_keys"] = [str(value) for value in feature_cache.keys()]
        action_history = state.get("action_history")
        if isinstance(action_history, Sequence) and not isinstance(action_history, (str, bytes)):
            summary["action_history"] = [
                {
                    "type": str(item.get("type")),
                    "frame_idx": item.get("frame_idx"),
                    "obj_ids": [int(value) for value in item.get("obj_ids") or []],
                }
                for item in action_history
                if isinstance(item, Mapping)
            ]
    return summary


def _runtime_version(model: Any) -> dict[str, Any]:
    try:
        sam3_version = importlib.metadata.version("sam3")
    except Exception:
        try:
            sam3_version = getattr(importlib.import_module("sam3"), "__version__", "unknown")
        except Exception:
            sam3_version = "unknown"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
    except Exception:
        torch_version = "unknown"
        cuda_version = "unknown"
    sam3_repo: str | None = None
    sam3_commit: str | None = None
    try:
        sam3_spec = importlib.util.find_spec("sam3")
        sam3_repo = _repo_path_from_spec(sam3_spec)
        if sam3_repo and (Path(sam3_repo) / ".git").exists():
            sam3_commit = subprocess.check_output(
                ["git", "-C", sam3_repo, "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
    except Exception:
        pass
    return {
        "sam3_version": str(sam3_version),
        "sam3_repo": sam3_repo,
        "sam3_git_commit": sam3_commit,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "python_version": platform.python_version(),
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def run_sam3_video_with_mask_prompt(
    video_info: Any,
    init_prompts: Iterable[Any],
    output_dir: str | Path,
    config: dict[str, Any],
) -> Sam3VideoResult:
    """Run SAM 3.1 with complete first-frame masks for all objects."""

    run_mode = str(config.get("sam3_run_mode", SAM3_RUN_MODE_FULL))
    if run_mode == SAM3_RUN_MODE_FULL:
        return _run_sam3_full_predictor_video_with_mask_prompt(video_info, init_prompts, output_dir, config)
    if run_mode == SAM3_RUN_MODE_LOW_LEVEL:
        return _run_sam3_low_level_video_with_mask_prompt(video_info, init_prompts, output_dir, config)
    return Sam3VideoResult(
        video_id=_video_id(video_info),
        status="failed",
        frame_count=0,
        error=f"Unsupported SAM 3.1 run mode: {run_mode}",
    )


def _init_full_predictor_mask_state(
    model: Any,
    state: Any,
    frame_idx: int,
    object_ids: list[int],
    mask_tensor: Any,
) -> dict[str, Any]:
    """Condition the full SAM 3.1 predictor with complete first-frame masks."""

    if not isinstance(state, dict):
        raise RuntimeError("SAM 3.1 full predictor init_state did not return a dictionary")
    world_size = int(getattr(model, "world_size", 1))
    rank = int(getattr(model, "rank", 0))
    if world_size != 1 or rank != 0:
        raise RuntimeError(f"SAM 3.1 mask-conditioning adapter currently requires single-GPU inference; got rank={rank}, world_size={world_size}")

    import torch
    import torch.nn.functional as F

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    masks = mask_tensor.float().to(device)
    if masks.ndim != 3:
        raise RuntimeError(f"First-frame masks must have shape [N,H,W], got {tuple(masks.shape)}")

    prepare = getattr(model, "_prepare_backbone_feats", None)
    if not callable(prepare):
        raise RuntimeError("SAM 3.1 full predictor is missing _prepare_backbone_feats; cannot build tracker feature cache")
    prepare(state, frame_idx, False)
    if not state.get("feature_cache"):
        raise RuntimeError("SAM 3.1 full predictor did not populate feature_cache before mask conditioning")

    state["sam2_inference_states"] = model._tracker_add_new_objects(
        frame_idx=frame_idx,
        num_frames=int(state["num_frames"]),
        new_obj_ids=[int(object_id) for object_id in object_ids],
        new_obj_masks=masks,
        tracker_states_local=state["sam2_inference_states"],
        orig_vid_height=int(state["orig_height"]),
        orig_vid_width=int(state["orig_width"]),
        feature_cache=state["feature_cache"],
    )
    if not state["sam2_inference_states"]:
        raise RuntimeError("SAM 3.1 full predictor did not create any tracker inference state from mask prompts")

    metadata = state["tracker_metadata"]
    metadata.clear()
    metadata.update(model._initialize_metadata())
    object_array = np.asarray(object_ids, dtype=np.int64)
    metadata["obj_ids_per_gpu"] = [object_array.copy()]
    metadata["obj_ids_all_gpu"] = object_array.copy()
    metadata["num_obj_per_gpu"] = np.asarray([len(object_ids)], dtype=np.int64)
    metadata["max_obj_id"] = int(max(object_ids))
    metadata["obj_id_to_score"] = {int(object_id): 1.0 for object_id in object_ids}
    score_key = "obj_id_to_sam2_score_frame_wise"
    metadata[score_key] = defaultdict(dict)
    metadata[score_key][frame_idx] = {
        int(object_id): torch.tensor(1.0, dtype=torch.float32, device=device)
        for object_id in object_ids
    }
    metadata["obj_id_to_last_occluded"] = {}

    rank0_metadata = metadata.get("rank0_metadata", {})
    rank0_metadata["obj_first_frame_idx"] = {int(object_id): int(frame_idx) for object_id in object_ids}
    rank0_metadata["unmatched_frame_inds"] = defaultdict(list)
    rank0_metadata["trk_keep_alive"] = defaultdict(int)
    rank0_metadata["overlap_pair_to_frame_inds"] = defaultdict(list)
    rank0_metadata["removed_obj_ids"] = set()
    rank0_metadata["suppressed_obj_ids"] = defaultdict(set)
    if "masklet_confirmation" in rank0_metadata or bool(getattr(model, "masklet_confirmation_enable", False)):
        confirmed_value = 2
        try:
            from sam3.model.sam3_multiplex_base import MaskletConfirmationStatus

            confirmed_value = int(MaskletConfirmationStatus.CONFIRMED.value)
        except Exception:
            pass
        rank0_metadata["masklet_confirmation"] = {
            "status": np.full(len(object_ids), confirmed_value, dtype=np.int64),
            "consecutive_det_num": np.full(
                len(object_ids),
                int(getattr(model, "masklet_confirmation_consecutive_det_thresh", 1)),
                dtype=np.int64,
            ),
        }
    metadata["rank0_metadata"] = rank0_metadata

    metadata["gpu_metadata"] = {
        "N_obj": len(object_ids),
        "obj_first_frame": torch.full((len(object_ids),), int(frame_idx), dtype=torch.long, device=device),
        "consecutive_unmatch_count": torch.zeros(len(object_ids), dtype=torch.long, device=device),
        "trk_keep_alive": torch.ones(len(object_ids), dtype=torch.bool, device=device),
        "removed_mask": torch.zeros(len(object_ids), dtype=torch.bool, device=device),
        "overlap_pair_counts": torch.zeros((len(object_ids), len(object_ids)), dtype=torch.long, device=device),
        "last_occluded_tensor": torch.zeros(len(object_ids), dtype=torch.long, device=device),
    }
    if bool(getattr(model, "is_multiplex", False)):
        count_buckets = getattr(model, "_count_buckets_in_states", None)
        num_buckets = int(count_buckets(state["sam2_inference_states"])) if callable(count_buckets) else len(state["sam2_inference_states"])
        metadata["num_buc_per_gpu"] = np.asarray([num_buckets], dtype=np.int64)

    video_masks = F.interpolate(
        masks.unsqueeze(1),
        size=(int(state["orig_height"]), int(state["orig_width"])),
        mode="nearest",
    ).to(torch.bool)
    obj_id_to_mask = {int(object_id): video_masks[index] for index, object_id in enumerate(object_ids)}
    model._cache_frame_outputs(state, frame_idx, obj_id_to_mask)

    model.add_action_history(state, "add", frame_idx=frame_idx, obj_ids=[int(object_id) for object_id in object_ids])
    if not state.get("action_history") or state["action_history"][-1]["type"] != "add":
        raise RuntimeError("SAM 3.1 full predictor did not record the mask add action")
    return {
        "mask_conditioning": "full_predictor_private_tracker_add_new_objects",
        "num_conditioned_objects": len(object_ids),
        "conditioned_object_ids": [int(object_id) for object_id in object_ids],
        "cached_conditioning_frames": [int(frame_idx)],
    }


def _set_full_predictor_tracking_bounds(state: Any, start_frame_idx: int, max_frame_num_to_track: int) -> None:
    if not isinstance(state, dict):
        return
    feature_cache = state.setdefault("feature_cache", {})
    if isinstance(feature_cache, dict):
        feature_cache["tracking_bounds"] = {
            "max_frame_num_to_track": int(max_frame_num_to_track),
            "propagate_in_video_start_frame_idx": int(start_frame_idx),
        }


def _frame_full_native_output(state: Any, frame_idx: int) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        return {}
    cached = state.get("cached_frame_outputs", {})
    if isinstance(cached, Mapping) and frame_idx in cached:
        return {"cached_frame_outputs": cached[frame_idx]}
    return {}


def _run_sam3_full_predictor_video_with_mask_prompt(
    video_info: Any,
    init_prompts: Iterable[Any],
    output_dir: str | Path,
    config: dict[str, Any],
) -> Sam3VideoResult:
    """Run the official full SAM 3.1 predictor with complete first-frame masks."""

    video_id = _video_id(video_info)
    output_root = Path(output_dir).expanduser().resolve()
    masks_dir = output_root / "masks" / video_id
    overlays_dir = output_root / "overlays" / video_id
    raw_logits_dir = output_root / "raw_logits" / video_id
    scores_path = output_root / "logs" / "native_scores" / f"{video_id}.jsonl"
    state_path = output_root / "native_state" / f"{video_id}.json"
    warnings: list[str] = []
    state: Any | None = None

    try:
        predictor = config["predictor"]
        _require_full_predictor_mask_api(predictor)
        model = _full_predictor_model(predictor)
        if str(config.get("prompt_mode", "mask")) != "mask":
            raise ValueError("SAM 3.1 full predictor baseline only supports --prompt-mode mask")
        if int(config.get("resize_long_side", 0) or 0) != 0:
            raise ValueError("SAM 3.1 full predictor baseline requires original resolution")

        prepared_frames = _prepare_frame_dir(video_info, config)
        max_frames = int(config.get("max_frames", 0) or 0)
        target_frame_count = min(len(prepared_frames), max_frames) if max_frames else len(prepared_frames)
        if target_frame_count <= 0:
            raise RuntimeError("No frames are available for SAM 3.1 inference")

        first = prepared_frames[0]
        data_root = Path(config["data_root"]).expanduser().resolve()
        original_annotation = _load_mask_prompt(init_prompts, data_root)
        if original_annotation.ndim == 3:
            original_annotation = np.any(original_annotation[..., :3] > 0, axis=-1).astype(np.uint8)
        original_annotation = _resize_mask(original_annotation, first.original_width, first.original_height)
        object_masks = _object_masks_from_initial_mask(original_annotation)
        object_ids = sorted(object_masks)
        if not object_ids:
            raise RuntimeError("The first-frame annotation contains no object IDs")

        frame_dir = _session_frame_dir(
            prepared_frames,
            target_frame_count,
            Path(config["cache_dir"]).expanduser().resolve(),
            video_id,
        )
        state = _initialize_native_state(
            model,
            frame_dir,
            offload_video=bool(config.get("offload_video_to_cpu", False)),
            offload_state=bool(config.get("offload_state_to_cpu", False)),
        )

        import torch

        mask_tensor = torch.from_numpy(np.stack([object_masks[object_id] for object_id in object_ids])).float()
        device = str(config.get("device", "cuda"))
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
        rows: list[dict[str, Any]] = []
        mask_paths: list[str] = []
        overlay_paths: list[str] = []
        raw_logit_paths: list[str] = []
        seen_frames: set[int] = set()
        previous_indexed: np.ndarray | None = None
        mask_conditioning_info: dict[str, Any] = {}
        diagnostics = _init_object_diagnostics(object_ids, target_frame_count, allow_missing_objects=True)

        with torch.inference_mode(), autocast:
            _set_full_predictor_tracking_bounds(state, 0, target_frame_count - 1)
            mask_conditioning_info = _init_full_predictor_mask_state(model, state, 0, object_ids, mask_tensor)
            _set_full_predictor_tracking_bounds(state, 0, target_frame_count - 1)
            propagation = _call_supported(
                model.propagate_in_video,
                inference_state=state,
                start_frame_idx=0,
                max_frame_num_to_track=target_frame_count - 1,
                reverse=False,
                output_prob_thresh=0.5,
                is_last_batch=True,
            )
            for item in propagation:
                frame_idx, output_ids, logits, object_scores = _normalize_propagation_item(item)
                if frame_idx < 0 or frame_idx >= target_frame_count:
                    continue
                if frame_idx in seen_frames:
                    raise RuntimeError(f"SAM 3.1 returned frame {frame_idx} more than once")
                raw_output_ids = list(output_ids)
                logits = _reorder_objects(logits, output_ids, object_ids, allow_missing=True)
                object_scores = _reorder_scores(object_scores, output_ids, object_ids, allow_missing=True)
                _record_object_diagnostics(diagnostics, frame_idx, object_ids, raw_output_ids, logits, object_scores)
                frame = prepared_frames[frame_idx]

                if frame_idx == 0:
                    indexed = original_annotation.astype(np.uint8, copy=True)
                else:
                    indexed = _compose_indexed_mask(logits, object_ids, previous_indexed)
                    indexed = _resize_mask(indexed, frame.original_width, frame.original_height)
                illegal = sorted(set(int(value) for value in np.unique(indexed)) - {0, *object_ids})
                if illegal:
                    raise RuntimeError(f"Frame {frame_idx} contains unexpected object IDs: {illegal}")

                mask_path = masks_dir / f"{frame.frame_stem}.png"
                overlay_path = overlays_dir / f"{frame.frame_stem}.jpg"
                _save_indexed_mask(indexed, mask_path)
                mask_paths.append(str(mask_path))
                if _should_save_overlay(frame_idx, config, warnings):
                    _save_overlay(Path(frame.original_path), indexed, overlay_path)
                    overlay_paths.append(str(overlay_path))
                previous_indexed = indexed
                seen_frames.add(frame_idx)

                rows.extend(_score_rows(video_id, frame_idx, object_ids, logits, object_scores, _frame_full_native_output(state, frame_idx)))
                if bool(config.get("save_raw_logits", False)):
                    raw_path = raw_logits_dir / f"{frame.frame_stem}.npz"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        raw_path,
                        object_ids=np.asarray(object_ids, dtype=np.int32),
                        mask_logits=logits.astype(np.float16),
                        object_score_logits=object_scores.astype(np.float32) if object_scores is not None else np.asarray([]),
                    )
                    raw_logit_paths.append(str(raw_path))

        expected_frames = set(range(target_frame_count))
        if seen_frames != expected_frames:
            missing = sorted(expected_frames - seen_frames)
            raise RuntimeError(f"SAM 3.1 propagation omitted {len(missing)} frames: {missing[:10]}")

        first_saved = np.asarray(Image.open(mask_paths[0]))
        first_frame_exact = bool(np.array_equal(first_saved, original_annotation.astype(np.uint8)))
        if not first_frame_exact:
            raise RuntimeError("Saved first-frame mask is not pixel-identical to the input annotation")

        diagnostics = _finalize_object_diagnostics(diagnostics)
        if diagnostics.get("total_missing_output_frames"):
            warnings.append(
                "The official SAM 3.1 full predictor omitted expected object IDs in "
                f"{diagnostics['total_missing_output_frames']} frame/object case(s); see diagnostics."
            )

        version = _runtime_version(model)
        if bool(config.get("save_native_scores", False)):
            if rows and all(row["predicted_iou"] is None for row in rows):
                warnings.append(
                    "The official SAM 3.1 full predictor did not expose predicted-IoU; "
                    "the JSONL field is null and inference behavior was left unchanged."
                )
            _write_jsonl(scores_path, rows)
            state_summary = _state_summary(state, object_ids, version)
            state_summary["mask_conditioning"] = mask_conditioning_info
            state_summary["backend_mode"] = SAM3_RUN_MODE_FULL
            state_summary["diagnostics"] = diagnostics
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state_summary, indent=2, ensure_ascii=True), encoding="utf-8")
        return Sam3VideoResult(
            video_id=video_id,
            status="done",
            frame_count=target_frame_count,
            object_ids=object_ids,
            mask_paths=mask_paths,
            overlay_paths=overlay_paths,
            raw_logit_paths=raw_logit_paths,
            native_scores_path=str(scores_path) if config.get("save_native_scores") else None,
            native_state_path=str(state_path) if config.get("save_native_scores") else None,
            warnings=warnings,
            first_frame_exact=True,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        return Sam3VideoResult(
            video_id=video_id,
            status="failed",
            frame_count=0,
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            sam3_available=True,
        )
    finally:
        if state is not None:
            reset = getattr(model if "model" in locals() else config.get("predictor"), "reset_state", None)
            if callable(reset):
                try:
                    reset(state)
                except Exception:
                    pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _run_sam3_low_level_video_with_mask_prompt(
    video_info: Any,
    init_prompts: Iterable[Any],
    output_dir: str | Path,
    config: dict[str, Any],
) -> Sam3VideoResult:
    """Run native SAM 3.1 with complete first-frame masks for all objects."""

    video_id = _video_id(video_info)
    output_root = Path(output_dir).expanduser().resolve()
    masks_dir = output_root / "masks" / video_id
    overlays_dir = output_root / "overlays" / video_id
    raw_logits_dir = output_root / "raw_logits" / video_id
    scores_path = output_root / "logs" / "native_scores" / f"{video_id}.jsonl"
    state_path = output_root / "native_state" / f"{video_id}.json"
    warnings: list[str] = []
    state: Any | None = None

    try:
        model = config["predictor"]
        _require_native_api(model)
        if str(config.get("prompt_mode", "mask")) != "mask":
            raise ValueError("SAM 3.1 baseline only supports --prompt-mode mask")
        if int(config.get("resize_long_side", 0) or 0) != 0:
            raise ValueError("SAM 3.1 native baseline requires original resolution")

        prepared_frames = _prepare_frame_dir(video_info, config)
        max_frames = int(config.get("max_frames", 0) or 0)
        target_frame_count = min(len(prepared_frames), max_frames) if max_frames else len(prepared_frames)
        if target_frame_count <= 0:
            raise RuntimeError("No frames are available for SAM 3.1 inference")

        first = prepared_frames[0]
        data_root = Path(config["data_root"]).expanduser().resolve()
        original_annotation = _load_mask_prompt(init_prompts, data_root)
        if original_annotation.ndim == 3:
            original_annotation = np.any(original_annotation[..., :3] > 0, axis=-1).astype(np.uint8)
        original_annotation = _resize_mask(original_annotation, first.original_width, first.original_height)
        object_masks = _object_masks_from_initial_mask(original_annotation)
        object_ids = sorted(object_masks)
        if not object_ids:
            raise RuntimeError("The first-frame annotation contains no object IDs")

        frame_dir = _session_frame_dir(
            prepared_frames,
            target_frame_count,
            Path(config["cache_dir"]).expanduser().resolve(),
            video_id,
        )
        state = _initialize_native_state(
            model,
            frame_dir,
            offload_video=bool(config.get("offload_video_to_cpu", False)),
            offload_state=bool(config.get("offload_state_to_cpu", False)),
        )

        import torch

        mask_tensor = torch.from_numpy(np.stack([object_masks[object_id] for object_id in object_ids])).bool()
        device = str(config.get("device", "cuda"))
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
        rows: list[dict[str, Any]] = []
        mask_paths: list[str] = []
        overlay_paths: list[str] = []
        raw_logit_paths: list[str] = []
        seen_frames: set[int] = set()
        previous_indexed: np.ndarray | None = None

        with torch.inference_mode(), autocast:
            model.add_new_masks(
                inference_state=state,
                frame_idx=0,
                obj_ids=object_ids,
                masks=mask_tensor,
                add_mask_to_memory=True,
            )
            propagation = model.propagate_in_video(
                inference_state=state,
                start_frame_idx=0,
                max_frame_num_to_track=target_frame_count - 1,
                reverse=False,
                tqdm_disable=bool(config.get("disable_tqdm", False)),
                run_mem_encoder=True,
            )
            for item in propagation:
                frame_idx, output_ids, logits, object_scores = _normalize_propagation_item(item)
                if frame_idx < 0 or frame_idx >= target_frame_count:
                    continue
                if frame_idx in seen_frames:
                    raise RuntimeError(f"SAM 3.1 returned frame {frame_idx} more than once")
                logits = _reorder_objects(logits, output_ids, object_ids)
                object_scores = _reorder_scores(object_scores, output_ids, object_ids)
                frame = prepared_frames[frame_idx]

                if frame_idx == 0:
                    indexed = original_annotation.astype(np.uint8, copy=True)
                else:
                    indexed = _compose_indexed_mask(logits, object_ids, previous_indexed)
                    indexed = _resize_mask(indexed, frame.original_width, frame.original_height)
                if frame_idx != 0:
                    illegal = sorted(set(int(value) for value in np.unique(indexed)) - {0, *object_ids})
                    if illegal:
                        raise RuntimeError(f"Frame {frame_idx} contains unexpected object IDs: {illegal}")

                mask_path = masks_dir / f"{frame.frame_stem}.png"
                overlay_path = overlays_dir / f"{frame.frame_stem}.jpg"
                _save_indexed_mask(indexed, mask_path)
                mask_paths.append(str(mask_path))
                if _should_save_overlay(frame_idx, config, warnings):
                    _save_overlay(Path(frame.original_path), indexed, overlay_path)
                    overlay_paths.append(str(overlay_path))
                previous_indexed = indexed
                seen_frames.add(frame_idx)

                native_output = _frame_native_output(state, frame_idx)
                rows.extend(_score_rows(video_id, frame_idx, object_ids, logits, object_scores, native_output))
                if bool(config.get("save_raw_logits", False)):
                    raw_path = raw_logits_dir / f"{frame.frame_stem}.npz"
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        raw_path,
                        object_ids=np.asarray(object_ids, dtype=np.int32),
                        mask_logits=logits.astype(np.float16),
                        object_score_logits=object_scores.astype(np.float32) if object_scores is not None else np.asarray([]),
                    )
                    raw_logit_paths.append(str(raw_path))

        expected_frames = set(range(target_frame_count))
        if seen_frames != expected_frames:
            missing = sorted(expected_frames - seen_frames)
            raise RuntimeError(f"SAM 3.1 propagation omitted {len(missing)} frames: {missing[:10]}")

        first_saved = np.asarray(Image.open(mask_paths[0]))
        first_frame_exact = bool(np.array_equal(first_saved, original_annotation.astype(np.uint8)))
        if not first_frame_exact:
            raise RuntimeError("Saved first-frame mask is not pixel-identical to the input annotation")

        version = _runtime_version(model)
        if bool(config.get("save_native_scores", False)):
            if rows and all(row["predicted_iou"] is None for row in rows):
                warnings.append(
                    "The official compact SAM 3.1 state did not expose predicted-IoU; "
                    "the JSONL field is null and inference behavior was left unchanged."
                )
            _write_jsonl(scores_path, rows)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(_state_summary(state, object_ids, version), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
        return Sam3VideoResult(
            video_id=video_id,
            status="done",
            frame_count=target_frame_count,
            object_ids=object_ids,
            mask_paths=mask_paths,
            overlay_paths=overlay_paths,
            raw_logit_paths=raw_logit_paths,
            native_scores_path=str(scores_path) if config.get("save_native_scores") else None,
            native_state_path=str(state_path) if config.get("save_native_scores") else None,
            warnings=warnings,
            first_frame_exact=True,
        )
    except Exception as exc:
        return Sam3VideoResult(
            video_id=video_id,
            status="failed",
            frame_count=0,
            warnings=warnings,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            sam3_available=True,
        )
    finally:
        if state is not None:
            reset = getattr(config.get("predictor"), "reset_state", None)
            if callable(reset):
                try:
                    reset(state)
                except Exception:
                    pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_sam3_video_with_multi_anchors(*args: Any, **kwargs: Any) -> Sam3VideoResult:
    """Reject the retired video-level pseudo-anchor path for SAM 3.1."""

    raise RuntimeError(
        "Video-level pseudo-anchor propagation is intentionally disabled for SAM 3.1. "
        "Use native first-frame mask conditioning and validate object-level recovery separately."
    )


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).lower() in {"1", "true", "yes", "on"}

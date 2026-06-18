"""DINO-style object crop features with automatic lightweight fallback."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.vos.reliability import extract_masked_feature


DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
DEFAULT_DINOV3_CANDIDATES = (
    "facebook/dinov3-base",
    "facebook/dinov3-vitb16",
    "facebook/dinov3-vits16",
)


@dataclass(slots=True)
class DinoFeatureBackend:
    """Feature backend metadata and optional loaded DINO model objects."""

    backend: str
    model_name: str
    device: str
    available: bool
    processor: Any | None = None
    model: Any | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe backend metadata."""

        payload = asdict(self)
        payload["processor"] = type(self.processor).__name__ if self.processor is not None else None
        payload["model"] = type(self.model).__name__ if self.model is not None else None
        return payload


@dataclass(slots=True)
class TargetFeaturePool:
    """Augmented target feature pool for object retrieval."""

    features: np.ndarray
    augmentations: list[str]
    backend: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding large feature arrays."""

        return {
            "num_features": int(self.features.shape[0]) if self.features.ndim == 2 else 0,
            "feature_dim": int(self.features.shape[1]) if self.features.ndim == 2 else 0,
            "augmentations": self.augmentations,
            "backend": self.backend,
        }


def _resolve_device(device: str | None) -> str:
    """Resolve requested device to cuda when available, otherwise cpu."""

    if device and device != "auto":
        if device.startswith("cuda"):
            try:
                import torch

                return device if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _candidate_model_names(model_name: str, fallback: bool) -> list[str]:
    """Return DINOv3-first model candidates followed by fallback model names."""

    env_candidates = [
        item.strip()
        for item in os.environ.get("DINO_V3_MODEL_CANDIDATES", "").split(",")
        if item.strip()
    ]
    candidates: list[str] = []
    for candidate in [*env_candidates, *DEFAULT_DINOV3_CANDIDATES, model_name]:
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    if fallback and DEFAULT_DINOV2_MODEL not in candidates:
        candidates.append(DEFAULT_DINOV2_MODEL)
    return candidates


def _lightweight_backend(model_name: str, device: str, warnings: list[str]) -> DinoFeatureBackend:
    """Build a lightweight fallback backend descriptor."""

    return DinoFeatureBackend(
        backend="lightweight",
        model_name=model_name,
        device=device,
        available=False,
        warnings=warnings,
    )


def build_dino_model(
    model_name: str = DEFAULT_DINOV2_MODEL,
    fallback: bool = True,
    device: str = "cuda",
) -> DinoFeatureBackend:
    """Build a DINOv3/DINOv2 feature backend, falling back without raising."""

    resolved_device = _resolve_device(device)
    warnings: list[str] = []
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except Exception as exc:
        warnings.append(f"transformers/torch import failed; using lightweight features: {type(exc).__name__}: {exc}")
        return _lightweight_backend(model_name, resolved_device, warnings)

    for candidate in _candidate_model_names(model_name, fallback):
        try:
            processor = AutoImageProcessor.from_pretrained(candidate)
            model = AutoModel.from_pretrained(candidate)
            model.eval()
            model.to(resolved_device)
            backend_name = "dinov3" if "dinov3" in candidate.lower() else "dinov2"
            return DinoFeatureBackend(
                backend=backend_name,
                model_name=candidate,
                device=resolved_device,
                available=True,
                processor=processor,
                model=model,
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"{candidate} unavailable; trying fallback: {type(exc).__name__}: {exc}")
            continue
    warnings.append("No DINO backend could be loaded; using lightweight features.")
    return _lightweight_backend(model_name, resolved_device, warnings)


def _as_rgb_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Convert image-like input to RGB PIL image."""

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    return Image.fromarray(array[..., :3].astype(np.uint8)).convert("RGB")


def _as_mask(mask: str | Path | Image.Image | np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Convert mask-like input to a boolean mask in image size."""

    if isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask))
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask)
    else:
        array = np.asarray(mask)
    if array.ndim == 3:
        array = np.any(array[..., :3] > 0, axis=-1)
    else:
        array = array > 0
    if array.shape != (size[1], size[0]):
        array = np.asarray(Image.fromarray(array.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0
    return array


def _box_from_mask(mask: np.ndarray) -> list[int] | None:
    """Return inclusive bbox for a mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_from_box(box: Any, size: tuple[int, int]) -> np.ndarray:
    """Create a boolean mask from an inclusive bbox."""

    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.size != 4:
        raise ValueError(f"mask_or_box must be a mask or 4-value bbox, got shape {values.shape}")
    width, height = size
    x0 = int(np.clip(np.floor(values[0]), 0, width - 1))
    y0 = int(np.clip(np.floor(values[1]), 0, height - 1))
    x1 = int(np.clip(np.ceil(values[2]), 0, width - 1))
    y1 = int(np.clip(np.ceil(values[3]), 0, height - 1))
    mask = np.zeros((height, width), dtype=bool)
    if x1 >= x0 and y1 >= y0:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _mask_or_box_to_mask(mask_or_box: Any, size: tuple[int, int]) -> np.ndarray:
    """Normalize mask-or-box input to boolean mask."""

    array = np.asarray(mask_or_box) if not isinstance(mask_or_box, (str, Path, Image.Image)) else None
    if array is not None and array.ndim == 1 and array.size == 4:
        return _mask_from_box(array, size)
    return _as_mask(mask_or_box, size)


def _crop_object(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    """Crop image and mask to the foreground bbox and zero background."""

    box = _box_from_mask(mask)
    if box is None:
        return image.copy(), np.zeros((image.height, image.width), dtype=bool)
    x0, y0, x1, y1 = box
    crop = image.crop((x0, y0, x1 + 1, y1 + 1))
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    crop_array = np.asarray(crop).copy()
    crop_array[~crop_mask] = 0
    return Image.fromarray(crop_array), crop_mask


def _normalize(feature: np.ndarray) -> np.ndarray:
    """L2-normalize a feature vector."""

    vector = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _dino_feature(image: Image.Image, backend: DinoFeatureBackend) -> np.ndarray:
    """Extract a DINO embedding from a PIL crop."""

    if not backend.available or backend.model is None or backend.processor is None:
        raise RuntimeError("DINO backend is not available.")
    import torch

    inputs = backend.processor(images=image, return_tensors="pt")
    inputs = {key: value.to(backend.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = backend.model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feature = outputs.pooler_output[0]
    elif hasattr(outputs, "last_hidden_state"):
        feature = outputs.last_hidden_state[:, 1:, :].mean(dim=1)[0]
    else:
        feature = outputs[0][:, 1:, :].mean(dim=1)[0]
    return _normalize(feature.detach().float().cpu().numpy())


def extract_crop_feature(
    image: str | Path | Image.Image | np.ndarray,
    mask_or_box: Any,
    backend: DinoFeatureBackend | None = None,
) -> np.ndarray:
    """Extract an object crop feature from a mask or bbox."""

    image_pil = _as_rgb_image(image)
    mask = _mask_or_box_to_mask(mask_or_box, image_pil.size)
    crop, crop_mask = _crop_object(image_pil, mask)
    if backend is None:
        backend = build_dino_model()
    if backend.available:
        try:
            return _dino_feature(crop, backend)
        except Exception as exc:
            backend.warnings.append(f"DINO feature extraction failed; using lightweight feature: {type(exc).__name__}: {exc}")
    return _normalize(extract_masked_feature(crop, crop_mask))


def _scale_image_and_mask(image: Image.Image, mask: np.ndarray, scale: float) -> tuple[Image.Image, np.ndarray]:
    """Scale an image and mask around their current crop."""

    width, height = image.size
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    scaled_image = image.resize(new_size, Image.Resampling.BILINEAR)
    scaled_mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(new_size, Image.Resampling.NEAREST)) > 0
    return scaled_image, scaled_mask


def _augment_crop(image: Image.Image, mask: np.ndarray, name: str) -> tuple[Image.Image, np.ndarray]:
    """Apply one target-pool augmentation to a crop and mask."""

    if name == "identity":
        return image.copy(), mask.copy()
    if name == "hflip":
        return ImageOps.mirror(image), np.fliplr(mask)
    if name == "rotate(-10)":
        return image.rotate(-10, resample=Image.Resampling.BILINEAR, expand=True), np.asarray(
            Image.fromarray(mask.astype(np.uint8)).rotate(-10, resample=Image.Resampling.NEAREST, expand=True)
        ) > 0
    if name == "rotate(+10)":
        return image.rotate(10, resample=Image.Resampling.BILINEAR, expand=True), np.asarray(
            Image.fromarray(mask.astype(np.uint8)).rotate(10, resample=Image.Resampling.NEAREST, expand=True)
        ) > 0
    if name == "scale(0.8)":
        return _scale_image_and_mask(image, mask, 0.8)
    if name == "scale(1.2)":
        return _scale_image_and_mask(image, mask, 1.2)
    raise ValueError(f"Unknown augmentation: {name}")


def extract_augmented_target_pool(
    image0: str | Path | Image.Image | np.ndarray,
    mask0: Any,
    backend: DinoFeatureBackend | None = None,
) -> TargetFeaturePool:
    """Extract an augmented normalized target feature pool for object retrieval."""

    image_pil = _as_rgb_image(image0)
    mask = _mask_or_box_to_mask(mask0, image_pil.size)
    crop, crop_mask = _crop_object(image_pil, mask)
    if backend is None:
        backend = build_dino_model()
    augmentation_names = ["identity", "hflip", "rotate(-10)", "rotate(+10)", "scale(0.8)", "scale(1.2)"]
    features: list[np.ndarray] = []
    used_augmentations: list[str] = []
    for name in augmentation_names:
        aug_image, aug_mask = _augment_crop(crop, crop_mask, name)
        features.append(extract_crop_feature(aug_image, aug_mask, backend))
        used_augmentations.append(name)
    feature_array = np.stack([_normalize(feature) for feature in features], axis=0).astype(np.float32)
    return TargetFeaturePool(
        features=feature_array,
        augmentations=used_augmentations,
        backend=backend.to_dict(),
    )

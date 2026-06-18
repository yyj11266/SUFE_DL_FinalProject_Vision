"""Optional SAM3 detector adapter for visual/text prompt candidate generation."""

from __future__ import annotations

import importlib
import inspect
import warnings as py_warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from src.trackers.sam3_tracker_optional import Sam3Availability, check_sam3_available
from src.vos.reliability import mask_area, mask_to_bbox


@dataclass(slots=True)
class Sam3DetectorCandidate:
    """SAM3 detector candidate compatible with anchor mining."""

    mask: np.ndarray
    bbox: list[float]
    confidence: float
    label: str = "sam3_candidate"
    source: str = "sam3"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary payload consumed by anchor mining."""

        return {
            "mask": self.mask.astype(bool),
            "bbox": [float(value) for value in self.bbox],
            "confidence": float(self.confidence),
            "score": float(self.confidence),
            "label": self.label,
            "source": self.source,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class Sam3DetectorBuildResult:
    """Result of building the optional SAM3 image processor."""

    available: bool
    processor: Any | None = None
    status: Sam3Availability | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe build metadata."""

        return {
            "available": self.available,
            "status": self.status.to_dict() if self.status is not None else None,
            "warnings": list(self.warnings),
            "error": self.error,
            "processor_built": self.processor is not None,
        }


@dataclass(slots=True)
class Sam3OptionalDetector:
    """Duck-typed SAM3 detector adapter for anchor mining."""

    checkpoint_path: str | Path | None = None
    device: str = "cuda"
    processor: Any | None = None
    label: str = "sam3"
    warnings: list[str] = field(default_factory=list)

    def build(self) -> Sam3DetectorBuildResult:
        """Build and cache the SAM3 image processor if available."""

        if self.processor is not None:
            return Sam3DetectorBuildResult(True, processor=self.processor, warnings=list(self.warnings))
        result = _build_sam3_image_processor(self.checkpoint_path, self.device)
        self.warnings.extend(result.warnings)
        if result.available:
            self.processor = result.processor
        return result

    def detect(
        self,
        image: Image.Image | np.ndarray | str | Path,
        frame_index: int | None = None,
        visual_prompt: Any = None,
        text_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Detect SAM3 candidates using visual prompt first, then text prompt."""

        del frame_index, kwargs
        result = self.build()
        if not result.available or result.processor is None:
            self.warnings.extend(result.warnings)
            _warn_once(f"SAM3 detector unavailable: {result.error or 'backend not available'}")
            return []
        candidates: list[Sam3DetectorCandidate] = []
        if visual_prompt is not None:
            candidates = run_sam3_detector_with_visual_prompt(
                image,
                visual_prompt,
                processor=result.processor,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        if not candidates and text_prompt:
            candidates = run_sam3_detector_with_text_prompt(
                image,
                text_prompt,
                processor=result.processor,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        return [candidate.to_dict() for candidate in candidates]

    def generate(self, image: Image.Image | np.ndarray | str | Path, **kwargs: Any) -> list[dict[str, Any]]:
        """Generate candidates through the same path as ``detect``."""

        return self.detect(image=image, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe adapter metadata."""

        return {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            "device": self.device,
            "processor": self.processor is not None,
            "label": self.label,
            "warnings": list(self.warnings),
        }


def run_sam3_detector_with_visual_prompt(
    frame: Image.Image | np.ndarray | str | Path,
    first_frame_crop_or_mask: Any,
    processor: Any | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
) -> list[Sam3DetectorCandidate]:
    """Run SAM3 detector with an exemplar/mask/box visual prompt."""

    built = Sam3DetectorBuildResult(True, processor=processor) if processor is not None else _build_sam3_image_processor(checkpoint_path, device)
    if not built.available or built.processor is None:
        _warn_once(f"SAM3 visual detector unavailable: {built.error or 'backend not available'}")
        return []
    image = _load_image(frame)
    try:
        state = built.processor.set_image(image) if hasattr(built.processor, "set_image") else None
        output = _call_visual_prompt(built.processor, state, first_frame_crop_or_mask)
        return _candidates_from_output(output, image.size, default_label="visual_prompt")
    except Exception as exc:
        _warn_once(f"SAM3 visual prompt detector failed: {type(exc).__name__}: {exc}")
        return []


def run_sam3_detector_with_text_prompt(
    frame: Image.Image | np.ndarray | str | Path,
    text: str,
    processor: Any | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cuda",
) -> list[Sam3DetectorCandidate]:
    """Run SAM3 detector with a text concept prompt."""

    built = Sam3DetectorBuildResult(True, processor=processor) if processor is not None else _build_sam3_image_processor(checkpoint_path, device)
    if not built.available or built.processor is None:
        _warn_once(f"SAM3 text detector unavailable: {built.error or 'backend not available'}")
        return []
    image = _load_image(frame)
    try:
        state = built.processor.set_image(image) if hasattr(built.processor, "set_image") else None
        if hasattr(built.processor, "set_text_prompt"):
            output = _call_with_supported_kwargs(built.processor.set_text_prompt, {"state": state, "prompt": text, "text": text})
        elif hasattr(built.processor, "detect"):
            output = _call_with_supported_kwargs(built.processor.detect, {"image": image, "text": text, "prompt": text})
        else:
            raise RuntimeError("SAM3 processor has no set_text_prompt/detect method.")
        return _candidates_from_output(output, image.size, default_label=text)
    except Exception as exc:
        _warn_once(f"SAM3 text prompt detector failed: {type(exc).__name__}: {exc}")
        return []


def _build_sam3_image_processor(
    checkpoint_path: str | Path | None,
    device: str,
) -> Sam3DetectorBuildResult:
    """Build SAM3 image model and processor when available."""

    status = check_sam3_available(checkpoint_path=checkpoint_path, device=device)
    warnings = list(status.warnings)
    if not status.sam3_importable or not status.image_builder_available:
        return Sam3DetectorBuildResult(False, status=status, warnings=warnings, error="SAM3 image builder unavailable.")
    try:
        builder_mod = importlib.import_module("sam3.model_builder")
        builder = getattr(builder_mod, "build_sam3_image_model")
        kwargs = _builder_kwargs(builder, checkpoint_path, device)
        model = builder(**kwargs)
        if hasattr(model, "to") and "device" not in kwargs:
            model = model.to(device)
        processor_cls = _import_processor_class()
        processor = processor_cls(model)
        return Sam3DetectorBuildResult(True, processor=processor, status=status, warnings=warnings)
    except Exception as exc:
        return Sam3DetectorBuildResult(False, status=status, warnings=warnings, error=f"{type(exc).__name__}: {exc}")


def _import_processor_class() -> Any:
    """Import the official SAM3 image processor class."""

    candidates = [
        ("sam3.model.sam3_image_processor", "Sam3Processor"),
        ("sam3.sam3_image_processor", "Sam3Processor"),
    ]
    errors: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            errors.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not import Sam3Processor. " + " | ".join(errors))


def _call_visual_prompt(processor: Any, state: Any, prompt: Any) -> Any:
    """Call a compatible SAM3 visual-prompt processor method."""

    visual = _visual_prompt_kwargs(prompt)
    method_names = [
        "set_visual_prompt",
        "set_image_prompt",
        "set_exemplar_prompt",
        "set_mask_prompt",
        "set_box_prompt",
        "detect",
    ]
    for method_name in method_names:
        if not hasattr(processor, method_name):
            continue
        method = getattr(processor, method_name)
        kwargs = {"state": state, **visual}
        if method_name == "detect":
            kwargs = {"state": state, **visual}
        try:
            return _call_with_supported_kwargs(method, kwargs)
        except TypeError:
            continue
    raise RuntimeError("SAM3 processor has no supported visual prompt method.")


def _visual_prompt_kwargs(prompt: Any) -> dict[str, Any]:
    """Convert crop/mask/box visual prompt input to common SAM3 kwargs."""

    if isinstance(prompt, Image.Image):
        return {"exemplar_image": prompt, "image": prompt}
    if isinstance(prompt, (str, Path)):
        image = Image.open(prompt).convert("RGB")
        return {"exemplar_image": image, "image": image}
    array = np.asarray(prompt) if prompt is not None else None
    if array is not None and array.ndim == 1 and array.size == 4:
        box = [float(value) for value in array.reshape(4).tolist()]
        return {"box": box, "bbox": box}
    if array is not None and array.ndim >= 2:
        mask = array > 0
        bbox = mask_to_bbox(mask)
        payload: dict[str, Any] = {"mask": mask.astype(np.uint8)}
        if bbox is not None:
            payload["box"] = [float(value) for value in bbox]
            payload["bbox"] = [float(value) for value in bbox]
        return payload
    raise ValueError("Unsupported SAM3 visual prompt type.")


def _candidates_from_output(
    output: Any,
    image_size: tuple[int, int],
    default_label: str,
) -> list[Sam3DetectorCandidate]:
    """Normalize SAM3 processor output to detector candidates."""

    if output is None:
        return []
    if isinstance(output, Mapping) and "candidates" in output:
        output = output["candidates"]
    if isinstance(output, Mapping):
        masks = output.get("masks", output.get("pred_masks", output.get("mask")))
        boxes = output.get("boxes", output.get("bboxes", output.get("bbox")))
        scores = output.get("scores", output.get("confidence", output.get("confidences")))
        labels = output.get("labels", output.get("label", default_label))
        return _candidate_arrays_to_list(masks, boxes, scores, labels, image_size, default_label)
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes)):
        candidates: list[Sam3DetectorCandidate] = []
        for item in output:
            if isinstance(item, Sam3DetectorCandidate):
                candidates.append(item)
            elif isinstance(item, Mapping):
                candidates.extend(_candidates_from_output(item, image_size, default_label))
        return candidates
    return []


def _candidate_arrays_to_list(
    masks: Any,
    boxes: Any,
    scores: Any,
    labels: Any,
    image_size: tuple[int, int],
    default_label: str,
) -> list[Sam3DetectorCandidate]:
    """Convert parallel mask/box/score arrays to candidate dataclasses."""

    mask_list = _as_mask_list(masks)
    box_list = _as_box_list(boxes)
    score_list = _as_score_list(scores, max(len(mask_list), len(box_list)))
    label_list = _as_label_list(labels, max(len(mask_list), len(box_list)), default_label)
    count = max(len(mask_list), len(box_list))
    candidates: list[Sam3DetectorCandidate] = []
    for index in range(count):
        mask = mask_list[index] if index < len(mask_list) else None
        bbox = box_list[index] if index < len(box_list) else None
        if mask is None and bbox is not None:
            mask = _box_mask(bbox, image_size)
        if bbox is None and mask is not None:
            bbox_int = mask_to_bbox(mask)
            bbox = [float(value) for value in bbox_int] if bbox_int is not None else None
        if mask is None or bbox is None or mask_area(mask) <= 0:
            continue
        clipped = _clip_box(bbox, image_size)
        candidates.append(
            Sam3DetectorCandidate(
                mask=_resize_mask(mask, image_size),
                bbox=clipped,
                confidence=score_list[index] if index < len(score_list) else 0.5,
                label=label_list[index] if index < len(label_list) else default_label,
                source="sam3",
                metadata={"candidate_index": index},
            )
        )
    return candidates


def _builder_kwargs(builder: Any, checkpoint_path: str | Path | None, device: str) -> dict[str, Any]:
    """Build compatible kwargs for changing SAM3 builder signatures."""

    kwargs: dict[str, Any] = {}
    try:
        params = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return kwargs
    if checkpoint_path is not None:
        for key in ("checkpoint_path", "ckpt_path", "checkpoint"):
            if key in params:
                kwargs[key] = str(Path(checkpoint_path).expanduser())
                break
    if "device" in params:
        kwargs["device"] = device
    if "load_from_HF" in params and checkpoint_path is None:
        kwargs["load_from_HF"] = True
    return kwargs


def _call_with_supported_kwargs(method: Any, kwargs: dict[str, Any]) -> Any:
    """Call a method with only kwargs accepted by its signature."""

    try:
        params = inspect.signature(method).parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
            return method(**kwargs)
        supported = {key: value for key, value in kwargs.items() if key in params and value is not None}
        return method(**supported)
    except (TypeError, ValueError):
        return method(**{key: value for key, value in kwargs.items() if value is not None})


def _load_image(frame: Image.Image | np.ndarray | str | Path) -> Image.Image:
    """Load a frame as RGB PIL image."""

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, (str, Path)):
        return Image.open(frame).convert("RGB")
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return Image.fromarray(array[..., :3].astype(np.uint8)).convert("RGB")


def _as_mask_list(masks: Any) -> list[np.ndarray]:
    """Normalize masks to a list of boolean arrays."""

    if masks is None:
        return []
    array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    array = np.squeeze(array)
    if array.ndim == 2:
        return [array > 0]
    if array.ndim == 3:
        return [array[index] > 0 for index in range(array.shape[0])]
    if array.ndim == 4:
        return [array[index, 0] > 0 for index in range(array.shape[0])]
    return []


def _as_box_list(boxes: Any) -> list[list[float]]:
    """Normalize boxes to ``[x0,y0,x1,y1]`` lists."""

    if boxes is None:
        return []
    array = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    array = np.asarray(array, dtype=np.float32).reshape(-1, 4)
    return [[float(value) for value in row.tolist()] for row in array]


def _as_score_list(scores: Any, count: int) -> list[float]:
    """Normalize scores to a list of floats."""

    if scores is None:
        return [0.5 for _ in range(count)]
    array = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    values = np.asarray(array, dtype=np.float32).reshape(-1).tolist()
    return [float(value) for value in values]


def _as_label_list(labels: Any, count: int, default: str) -> list[str]:
    """Normalize labels to a list of strings."""

    if labels is None:
        return [default for _ in range(count)]
    if isinstance(labels, str):
        return [labels for _ in range(count)]
    try:
        values = list(labels)
    except TypeError:
        return [str(labels) for _ in range(count)]
    return [str(value) for value in values]


def _resize_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a mask to image size ``(width, height)``."""

    width, height = int(image_size[0]), int(image_size[1])
    array = np.asarray(mask, dtype=np.uint8)
    if array.shape[:2] == (height, width):
        return array > 0
    return np.asarray(Image.fromarray(array).resize((width, height), Image.Resampling.NEAREST)) > 0


def _box_mask(box: Sequence[float], image_size: tuple[int, int]) -> np.ndarray:
    """Create a rectangular mask from a bbox."""

    width, height = int(image_size[0]), int(image_size[1])
    x0, y0, x1, y1 = [int(round(value)) for value in _clip_box(box, image_size)]
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _clip_box(box: Sequence[float], image_size: tuple[int, int]) -> list[float]:
    """Clip bbox coordinates to image bounds."""

    width, height = int(image_size[0]), int(image_size[1])
    values = np.asarray(box, dtype=np.float32).reshape(4)
    values[[0, 2]] = np.clip(values[[0, 2]], 0, max(0, width - 1))
    values[[1, 3]] = np.clip(values[[1, 3]], 0, max(0, height - 1))
    if values[2] < values[0]:
        values[0], values[2] = values[2], values[0]
    if values[3] < values[1]:
        values[1], values[3] = values[3], values[1]
    return [float(value) for value in values.tolist()]


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-safe debug payloads."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return str(value)


def _warn_once(message: str) -> None:
    """Emit a runtime warning for optional SAM3 failures."""

    py_warnings.warn(message, RuntimeWarning, stacklevel=2)

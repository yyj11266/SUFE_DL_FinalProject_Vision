"""Optional SUTrack wrapper with a stable Kalman bbox fallback."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(slots=True)
class TrackResult:
    """Tracking result for one frame."""

    frame_index: int
    bbox: list[float]
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable tracking result."""

        return asdict(self)


@dataclass(slots=True)
class KalmanBBoxTracker:
    """Constant-velocity bbox tracker used when SUTrack is unavailable."""

    process_noise: float = 1.0
    measurement_noise: float = 4.0
    base_confidence: float = 0.60
    source: str = "kalman_bbox"
    warnings: list[str] = field(default_factory=list)
    initialized: bool = False
    state: np.ndarray = field(default_factory=lambda: np.zeros((8,), dtype=np.float32))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(8, dtype=np.float32) * 10.0)
    frame_index: int = -1

    def initialize(self, first_frame: Any, bbox: list[float] | tuple[float, float, float, float] | np.ndarray) -> None:
        """Initialize tracker state from first frame and bbox."""

        values = np.asarray(bbox, dtype=np.float32).reshape(4)
        self.state = np.zeros((8,), dtype=np.float32)
        self.state[:4] = values
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self.frame_index = 0
        self.initialized = True

    def _transition(self) -> np.ndarray:
        """Return constant-velocity transition matrix."""

        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = 1.0
        transition[1, 5] = 1.0
        transition[2, 6] = 1.0
        transition[3, 7] = 1.0
        return transition

    def _predict_state(self) -> None:
        """Advance tracker state one step."""

        transition = self._transition()
        q = np.eye(8, dtype=np.float32) * float(self.process_noise)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + q
        self.frame_index += 1

    def _clip_bbox(self, frame: Any) -> list[float]:
        """Clip current bbox to frame bounds when available."""

        bbox = self.state[:4].astype(np.float32)
        size = _frame_size(frame)
        if size is not None:
            width, height = size
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
        if bbox[2] < bbox[0]:
            bbox[2] = bbox[0]
        if bbox[3] < bbox[1]:
            bbox[3] = bbox[1]
        return [float(value) for value in bbox.tolist()]

    def track(self, frame: Any) -> TrackResult:
        """Predict bbox for the next frame."""

        if not self.initialized:
            raise RuntimeError("KalmanBBoxTracker must be initialized before track().")
        self._predict_state()
        confidence = max(0.05, float(self.base_confidence * np.exp(-0.015 * max(0, self.frame_index - 1))))
        return TrackResult(
            frame_index=self.frame_index,
            bbox=self._clip_bbox(frame),
            confidence=confidence,
            source=self.source,
        )

    def update(
        self,
        frame: Any,
        observed_bbox: list[float] | tuple[float, float, float, float] | np.ndarray,
        confidence: float = 1.0,
    ) -> TrackResult:
        """Update tracker with an observed bbox measurement."""

        if not self.initialized:
            self.initialize(frame, observed_bbox)
            return TrackResult(0, self._clip_bbox(frame), float(confidence), self.source)
        measurement = np.asarray(observed_bbox, dtype=np.float32).reshape(4)
        h = np.zeros((4, 8), dtype=np.float32)
        h[:, :4] = np.eye(4, dtype=np.float32)
        r_scale = max(0.1, 1.0 / max(0.05, float(confidence)))
        r = np.eye(4, dtype=np.float32) * float(self.measurement_noise * r_scale)
        y = measurement - h @ self.state
        s = h @ self.covariance @ h.T + r
        k = self.covariance @ h.T @ np.linalg.pinv(s)
        self.state = self.state + k @ y
        self.covariance = (np.eye(8, dtype=np.float32) - k @ h) @ self.covariance
        return TrackResult(self.frame_index, self._clip_bbox(frame), float(confidence), self.source)


class _SutrackAdapter:
    """Small duck-typed adapter around an externally installed SUTrack tracker."""

    def __init__(self, tracker: Any) -> None:
        """Store external tracker object."""

        self.tracker = tracker
        self.source = "sutrack"
        self.frame_index = 0
        self.warnings: list[str] = []

    def initialize(self, first_frame: Any, bbox: list[float] | tuple[float, float, float, float] | np.ndarray) -> None:
        """Initialize external SUTrack tracker."""

        if hasattr(self.tracker, "initialize"):
            self.tracker.initialize(first_frame, bbox)
        elif hasattr(self.tracker, "init"):
            self.tracker.init(first_frame, bbox)
        else:
            raise RuntimeError("SUTrack object has no initialize/init method.")
        self.frame_index = 0

    def track(self, frame: Any) -> TrackResult:
        """Track one frame with external SUTrack object."""

        if hasattr(self.tracker, "track"):
            payload = self.tracker.track(frame)
        else:
            raise RuntimeError("SUTrack object has no track method.")
        self.frame_index += 1
        bbox, confidence = _parse_external_track_payload(payload)
        return TrackResult(self.frame_index, bbox, confidence, self.source)

    def update(self, frame: Any, observed_bbox: Any, confidence: float = 1.0) -> TrackResult:
        """Update external tracker when supported."""

        if hasattr(self.tracker, "update"):
            self.tracker.update(frame, observed_bbox, confidence)
        return TrackResult(self.frame_index, [float(value) for value in np.asarray(observed_bbox).reshape(4).tolist()], float(confidence), self.source)


def _frame_size(frame: Any) -> tuple[int, int] | None:
    """Return frame size as ``(width, height)`` when known."""

    if isinstance(frame, (str, Path)):
        with Image.open(frame) as image:
            return image.size
    if isinstance(frame, Image.Image):
        return frame.size
    array = np.asarray(frame) if frame is not None else None
    if array is not None and array.ndim >= 2:
        return int(array.shape[1]), int(array.shape[0])
    return None


def _parse_external_track_payload(payload: Any) -> tuple[list[float], float]:
    """Parse external tracker output into bbox and confidence."""

    if isinstance(payload, dict):
        bbox = payload.get("bbox", payload.get("box", payload.get("target_bbox")))
        confidence = payload.get("confidence", payload.get("score", 0.75))
    elif isinstance(payload, (list, tuple)) and len(payload) >= 2:
        bbox, confidence = payload[0], payload[1]
    else:
        bbox, confidence = payload, 0.75
    values = np.asarray(bbox, dtype=np.float32).reshape(4)
    return [float(value) for value in values.tolist()], float(confidence)


def build_sutrack_or_fallback(**kwargs: Any) -> Any:
    """Build SUTrack if importable, otherwise return ``KalmanBBoxTracker``."""

    warnings: list[str] = []
    for module_name in ("sutrack", "SUTrack", "lib.test.tracker.sutrack"):
        try:
            module = importlib.import_module(module_name)
            builder = getattr(module, "build_sutrack", None) or getattr(module, "create_tracker", None)
            if builder is None:
                tracker_cls = getattr(module, "SUTrack", None)
                tracker = tracker_cls(**kwargs) if tracker_cls is not None else None
            else:
                tracker = builder(**kwargs)
            if tracker is not None:
                return _SutrackAdapter(tracker)
        except Exception as exc:
            warnings.append(f"{module_name} unavailable: {type(exc).__name__}: {exc}")
    fallback = KalmanBBoxTracker()
    fallback.warnings.extend(warnings)
    fallback.warnings.append("SUTrack unavailable; using KalmanBBoxTracker fallback.")
    return fallback


def track_video_bboxes(
    frames: Iterable[Any],
    init_bbox: list[float] | tuple[float, float, float, float] | np.ndarray,
    tracker: Any | None = None,
) -> list[TrackResult]:
    """Track bboxes over a frame sequence with SUTrack or Kalman fallback."""

    frame_list = list(frames)
    if not frame_list:
        return []
    tracker_obj = tracker if tracker is not None else build_sutrack_or_fallback()
    tracker_obj.initialize(frame_list[0], init_bbox)
    results = [TrackResult(0, [float(value) for value in np.asarray(init_bbox, dtype=np.float32).reshape(4).tolist()], 1.0, getattr(tracker_obj, "source", "tracker"))]
    for frame in frame_list[1:]:
        results.append(tracker_obj.track(frame))
    return results

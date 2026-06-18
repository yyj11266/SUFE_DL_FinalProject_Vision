"""Object-level recovery state and delayed memory-admission policy.

This module is backend-agnostic. It does not alter masks by itself; callers use
its decisions to pause ordinary memory writes and manage bounded recovery
branches after the native SAM 3.1 baseline has passed holdout validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RecoveryState(str, Enum):
    STABLE = "stable"
    AMBIGUOUS = "ambiguous"
    LOST = "lost"
    RECOVERED = "recovered"


@dataclass(slots=True)
class RecoveryConfig:
    """Thresholds calibrated only on an external calibration split."""

    presence_stable: float = 0.65
    presence_lost: float = 0.20
    predicted_iou_stable: float = 0.65
    predicted_iou_ambiguous: float = 0.40
    appearance_stable: float = 0.70
    appearance_ambiguous: float = 0.50
    max_area_ratio_change: float = 3.0
    max_confusion_iou: float = 0.35
    lost_patience: int = 2
    recovery_confirm_frames: int = 2
    max_branches: int = 3


@dataclass(slots=True)
class RecoveryObservation:
    """Per-object evidence for one frame."""

    frame_index: int
    presence: float | None
    predicted_iou: float | None
    appearance_similarity: float | None
    area_ratio_change: float | None
    confusion_iou: float | None
    foreground_pixels: int


@dataclass(slots=True)
class RecoveryBranch:
    """One candidate identity/location branch during recovery."""

    branch_id: str
    source: str
    score: float
    frame_index: int
    identity_similarity: float
    position_consistency: float
    consecutive_confirmations: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_score(self) -> float:
        return 0.45 * self.score + 0.35 * self.identity_similarity + 0.20 * self.position_consistency


@dataclass(slots=True)
class RecoveryDecision:
    """State transition and memory/re-prompt actions for one frame."""

    object_id: int
    frame_index: int
    previous_state: RecoveryState
    state: RecoveryState
    allow_memory_write: bool
    keep_first_frame_memory: bool
    request_recovery: bool
    reason: str
    active_branch_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["previous_state"] = self.previous_state.value
        payload["state"] = self.state.value
        return payload


class ObjectRecoveryController:
    """Maintain one object's recovery state and at most three candidates."""

    def __init__(self, object_id: int, config: RecoveryConfig | None = None) -> None:
        self.object_id = int(object_id)
        self.config = config or RecoveryConfig()
        self.state = RecoveryState.STABLE
        self.low_confidence_streak = 0
        self.recovered_streak = 0
        self.branches: dict[str, RecoveryBranch] = {}
        self.confirmed_memory_frames: list[int] = [0]

    def add_branch(self, branch: RecoveryBranch) -> None:
        """Insert a candidate and retain only the strongest bounded set."""

        self.branches[branch.branch_id] = branch
        ordered = sorted(self.branches.values(), key=lambda item: (-item.rank_score, item.branch_id))
        self.branches = {branch.branch_id: branch for branch in ordered[: self.config.max_branches]}

    def update_branch(
        self,
        branch_id: str,
        *,
        frame_index: int,
        score: float,
        identity_similarity: float,
        position_consistency: float,
    ) -> bool:
        """Update consistency and return whether the branch is confirmed."""

        branch = self.branches[branch_id]
        consistent = identity_similarity >= self.config.appearance_stable and position_consistency >= 0.6
        branch.consecutive_confirmations = branch.consecutive_confirmations + 1 if consistent else 0
        branch.frame_index = int(frame_index)
        branch.score = float(score)
        branch.identity_similarity = float(identity_similarity)
        branch.position_consistency = float(position_consistency)
        return branch.consecutive_confirmations >= self.config.recovery_confirm_frames

    def _weak_reasons(self, observation: RecoveryObservation) -> list[str]:
        reasons: list[str] = []
        if observation.foreground_pixels <= 0:
            reasons.append("empty_mask")
        if observation.presence is not None and observation.presence < self.config.presence_stable:
            reasons.append("low_presence")
        if observation.predicted_iou is not None and observation.predicted_iou < self.config.predicted_iou_stable:
            reasons.append("low_predicted_iou")
        if observation.appearance_similarity is not None and observation.appearance_similarity < self.config.appearance_stable:
            reasons.append("appearance_drift")
        if observation.area_ratio_change is not None and observation.area_ratio_change > self.config.max_area_ratio_change:
            reasons.append("area_jump")
        if observation.confusion_iou is not None and observation.confusion_iou > self.config.max_confusion_iou:
            reasons.append("target_confusion")
        return reasons

    def step(self, observation: RecoveryObservation, confirmed_branch_id: str | None = None) -> RecoveryDecision:
        """Advance the state without admitting ambiguous frames to memory."""

        previous = self.state
        weak_reasons = self._weak_reasons(observation)
        explicitly_lost = (
            observation.foreground_pixels <= 0
            or (observation.presence is not None and observation.presence < self.config.presence_lost)
        )

        if confirmed_branch_id is not None:
            if confirmed_branch_id not in self.branches:
                raise KeyError(f"Unknown recovery branch: {confirmed_branch_id}")
            self.recovered_streak += 1
            self.low_confidence_streak = 0
            self.state = RecoveryState.RECOVERED
        elif explicitly_lost:
            self.low_confidence_streak += 1
            self.recovered_streak = 0
            self.state = (
                RecoveryState.LOST
                if self.low_confidence_streak >= self.config.lost_patience
                else RecoveryState.AMBIGUOUS
            )
        elif weak_reasons:
            self.low_confidence_streak += 1
            self.recovered_streak = 0
            self.state = RecoveryState.AMBIGUOUS
        else:
            self.low_confidence_streak = 0
            if previous == RecoveryState.RECOVERED:
                self.recovered_streak += 1
                self.state = (
                    RecoveryState.STABLE
                    if self.recovered_streak >= self.config.recovery_confirm_frames
                    else RecoveryState.RECOVERED
                )
            else:
                self.recovered_streak = 0
                self.state = RecoveryState.STABLE

        allow_memory_write = self.state in {RecoveryState.STABLE, RecoveryState.RECOVERED}
        if allow_memory_write and observation.frame_index not in self.confirmed_memory_frames:
            self.confirmed_memory_frames.append(observation.frame_index)
        request_recovery = self.state in {RecoveryState.AMBIGUOUS, RecoveryState.LOST}
        reason = ",".join(weak_reasons) if weak_reasons else (
            f"confirmed_branch={confirmed_branch_id}" if confirmed_branch_id else "quality_stable"
        )
        return RecoveryDecision(
            object_id=self.object_id,
            frame_index=observation.frame_index,
            previous_state=previous,
            state=self.state,
            allow_memory_write=allow_memory_write,
            keep_first_frame_memory=True,
            request_recovery=request_recovery,
            reason=reason,
            active_branch_ids=list(self.branches),
        )


class MultiObjectRecoveryManager:
    """Keep recovery state isolated for each object in a video."""

    def __init__(self, object_ids: list[int], config: RecoveryConfig | None = None) -> None:
        shared_config = config or RecoveryConfig()
        self.controllers = {
            int(object_id): ObjectRecoveryController(int(object_id), shared_config)
            for object_id in object_ids
        }

    def step(
        self,
        object_id: int,
        observation: RecoveryObservation,
        confirmed_branch_id: str | None = None,
    ) -> RecoveryDecision:
        return self.controllers[int(object_id)].step(observation, confirmed_branch_id)


"""Probe SAM 3.1 high-level video predictor state.

This script is intentionally diagnostic-only. It builds the full SAM 3.1
multiplex video predictor expected by the released merged checkpoint, audits
checkpoint coverage against the final model, and checks the public high-level
session API with explicit object IDs from first-frame point prompts.

The low-level add_new_masks path can be enabled as a research probe, but it is
not treated as a supported submission API. This script never creates masks or a
submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import sys
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.inspect_sufe import inspect_dataset
from src.trackers.sam2_tracker import (
    _load_mask_prompt,
    _object_masks_from_initial_mask,
    _prepare_frame_dir,
    _resize_mask,
)
from src.trackers.sam3_tracker_optional import (
    SAM31_CHECKPOINT_NAME,
    SAM31_HF_REPO,
    SAM3_RUN_MODE_OFFICIAL,
    _call_supported,
    _initialize_native_state,
    _normalize_propagation_item,
    _reorder_objects,
    _reorder_scores,
    _runtime_version,
    _session_frame_dir,
    _sigmoid,
    _state_summary,
    build_sam3_tracker,
    install_sam3_if_requested,
)


API_KEYWORDS = (
    "add",
    "annotation",
    "box",
    "cache",
    "condition",
    "history",
    "init",
    "mask",
    "point",
    "propagate",
    "reset",
    "state",
    "track",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug SAM 3.1 full predictor and public session state.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", default=f"sam31_api_probe_{dt.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--video-id", default="2b827e3a")
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--checkpoint")
    parser.add_argument("--hf-repo-id", default=SAM31_HF_REPO)
    parser.add_argument("--checkpoint-filename", default=SAM31_CHECKPOINT_NAME)
    parser.add_argument("--sam3-repo-dir")
    parser.add_argument("--install-sam3", action="store_true")
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--use-rope-real", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--allow-unsupported-runtime", action="store_true")
    parser.add_argument(
        "--run-low-level-mask-probe",
        action="store_true",
        help="Also try the low-level add_new_masks state probe after the full predictor probe.",
    )
    return parser


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _safe_signature(value: Any) -> str | None:
    if not callable(value):
        return None
    try:
        return str(inspect.signature(value))
    except Exception as exc:
        return f"<signature unavailable: {type(exc).__name__}: {exc}>"


def _safe_doc(value: Any) -> str | None:
    doc = inspect.getdoc(value)
    if not doc:
        return None
    return doc.splitlines()[0][:240]


def _callable_inventory(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(dir(obj)):
        if name.startswith("__") and name.endswith("__"):
            continue
        lowered = name.lower()
        if not any(keyword in lowered for keyword in API_KEYWORDS):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:
            rows.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if callable(value):
            rows.append(
                {
                    "name": name,
                    "signature": _safe_signature(value),
                    "doc": _safe_doc(value),
                    "defined_on": f"{type(value).__module__}.{type(value).__name__}",
                }
            )
    return rows


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return {"type": type(value).__name__, "truncated": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        payload: dict[str, Any] = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size <= 32:
            payload["values"] = value.tolist()
        elif np.issubdtype(value.dtype, np.number):
            payload["min"] = float(np.nanmin(value))
            payload["max"] = float(np.nanmax(value))
            payload["mean"] = float(np.nanmean(value))
        return payload
    if hasattr(value, "detach") and hasattr(value, "shape"):
        try:
            array = value.detach().float().cpu().numpy()
            payload = _json_safe(array, depth + 1)
            if isinstance(payload, dict):
                payload["type"] = "tensor"
                payload["device"] = str(getattr(value, "device", "unknown"))
            return payload
        except Exception:
            return {
                "type": "tensor",
                "shape": [int(dim) for dim in getattr(value, "shape", [])],
                "device": str(getattr(value, "device", "unknown")),
            }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {"type": type(value).__name__, "size": len(value)}
        items: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 40:
                items["..."] = f"{len(value) - index} more"
                break
            items[str(key)] = _json_safe(item, depth + 1)
        result["items"] = items
        return result
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item, depth + 1) for item in sorted(value, key=str)[:40]]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {
            "type": type(value).__name__,
            "size": len(value),
            "items": [_json_safe(item, depth + 1) for item in list(value)[:40]],
        }
    return {"type": type(value).__module__ + "." + type(value).__name__, "repr": repr(value)[:240]}


def _state_snapshot(state: Any, object_ids: list[int], model: Any, label: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "label": label,
        "summary": _state_summary(state, object_ids, _runtime_version(model)),
    }
    if isinstance(state, Mapping):
        snapshot["top_level_keys"] = sorted(str(key) for key in state.keys())
        for key in (
            "num_frames",
            "orig_height",
            "orig_width",
            "tracker_metadata",
            "sam2_inference_states",
            "cached_frame_outputs",
            "action_history",
            "feature_cache",
        ):
            if key in state:
                snapshot[key] = _json_safe(state[key])
    return snapshot


def _resolve_checkpoint(args: argparse.Namespace, exp_dir: Path) -> Path | None:
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        return path.resolve() if path.exists() else path
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=args.hf_repo_id,
            filename=args.checkpoint_filename,
            token=token,
            local_dir=str(checkpoint_dir),
        )
    ).resolve()


def _select_video(data_root: Path, video_id: str) -> Any:
    data_info = inspect_dataset(data_root)
    by_id = {video.video_id: video for video in data_info.videos}
    if video_id not in by_id:
        raise RuntimeError(f"Video ID {video_id!r} not found. Available examples: {sorted(by_id)[:10]}")
    return by_id[video_id]


def _key_prefix_summary(keys: list[str], sample_limit: int = 80) -> dict[str, Any]:
    one_part = Counter(key.split(".")[0] for key in keys)
    two_part = Counter(".".join(key.split(".")[:2]) for key in keys)
    return {
        "state_key_count": len(keys),
        "state_key_examples": keys[:sample_limit],
        "top_prefix_counts": one_part.most_common(40),
        "top_two_part_prefix_counts": two_part.most_common(60),
    }


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except Exception:
        return None


def _same_shape(source: Any, target: Any) -> bool:
    source_shape = _shape_tuple(source)
    target_shape = _shape_tuple(target)
    if source_shape is None or target_shape is None:
        return True
    return source_shape == target_shape


def _model_state_key_summary(model: Any) -> dict[str, Any]:
    try:
        state = model.state_dict()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    keys = [str(key) for key in state.keys()]
    summary = _key_prefix_summary(keys)
    for prefix in (
        "tracker.",
        "model.",
        "sam2_predictor.",
        "detector.",
        "backbone.",
        "maskmem_backbone.",
        "sam_prompt_encoder.",
        "sam_mask_decoder.",
        "transformer.",
        "segmentation_head.",
    ):
        summary[f"has_prefix:{prefix}"] = any(key.startswith(prefix) for key in keys)
    for key in (
        "maskmem_tpos_enc",
        "interactivity_no_mem_embed",
        "no_obj_embed_spatial",
        "output_valid_embed",
        "output_invalid_embed",
    ):
        summary[f"has_key:{key}"] = key in state
    return summary


def _checkpoint_model_coverage(state: Mapping[str, Any], model: Any) -> dict[str, Any]:
    try:
        model_state = model.state_dict()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    def identity(key: str) -> str | None:
        return key

    def strip_tracker_model(key: str) -> str | None:
        return key[len("tracker.model.") :] if key.startswith("tracker.model.") else None

    def strip_sam2_predictor_model(key: str) -> str | None:
        return key[len("sam2_predictor.model.") :] if key.startswith("sam2_predictor.model.") else None

    def strip_sam2_predictor(key: str) -> str | None:
        return key[len("sam2_predictor.") :] if key.startswith("sam2_predictor.") else None

    def detector_backbone_to_backbone(key: str) -> str | None:
        prefix = "detector.backbone.vision_backbone."
        if key.startswith(prefix):
            return "backbone.vision_backbone." + key[len(prefix) :]
        return None

    def strip_detector(key: str) -> str | None:
        return key[len("detector.") :] if key.startswith("detector.") else None

    transforms = (
        ("identity", identity),
        ("strip_tracker_model", strip_tracker_model),
        ("strip_sam2_predictor_model", strip_sam2_predictor_model),
        ("strip_sam2_predictor", strip_sam2_predictor),
        ("detector_backbone_to_backbone", detector_backbone_to_backbone),
        ("strip_detector", strip_detector),
    )

    model_keys = set(str(key) for key in model_state.keys())
    checkpoint_items = [(str(key), value) for key, value in state.items()]
    transform_rows: list[dict[str, Any]] = []
    for name, transform in transforms:
        candidates = 0
        key_matches = 0
        shape_matches = 0
        shape_mismatches = 0
        matched_examples: list[dict[str, str]] = []
        mismatch_examples: list[dict[str, Any]] = []
        for source_key, source_value in checkpoint_items:
            target_key = transform(source_key)
            if target_key is None:
                continue
            candidates += 1
            if target_key not in model_keys:
                continue
            key_matches += 1
            target_value = model_state[target_key]
            if _same_shape(source_value, target_value):
                shape_matches += 1
                if len(matched_examples) < 20:
                    matched_examples.append({"source": source_key, "target": target_key})
            else:
                shape_mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "source": source_key,
                            "target": target_key,
                            "source_shape": list(_shape_tuple(source_value) or ()),
                            "target_shape": list(_shape_tuple(target_value) or ()),
                        }
                    )
        transform_rows.append(
            {
                "transform": name,
                "candidate_checkpoint_keys": candidates,
                "model_key_matches": key_matches,
                "shape_matches": shape_matches,
                "shape_mismatches": shape_mismatches,
                "matched_examples": matched_examples,
                "mismatch_examples": mismatch_examples,
            }
        )

    priority = (
        ("identity", identity),
        ("strip_tracker_model", strip_tracker_model),
        ("strip_sam2_predictor_model", strip_sam2_predictor_model),
        ("strip_sam2_predictor", strip_sam2_predictor),
        ("detector_backbone_to_backbone", detector_backbone_to_backbone),
        ("strip_detector", strip_detector),
    )
    covered_targets: dict[str, str] = {}
    covered_sources: dict[str, str] = {}
    source_prefix_counts: Counter[str] = Counter()
    for source_key, source_value in checkpoint_items:
        for name, transform in priority:
            target_key = transform(source_key)
            if target_key is None or target_key not in model_keys:
                continue
            if not _same_shape(source_value, model_state[target_key]):
                continue
            covered_targets.setdefault(target_key, name)
            covered_sources[source_key] = name
            if source_key.startswith("tracker.model."):
                source_prefix_counts["tracker.model"] += 1
            elif source_key.startswith("detector.backbone.vision_backbone."):
                source_prefix_counts["detector.backbone.vision_backbone"] += 1
            elif source_key.startswith("detector."):
                source_prefix_counts["detector"] += 1
            elif source_key.startswith("sam2_predictor."):
                source_prefix_counts["sam2_predictor"] += 1
            else:
                source_prefix_counts["direct_or_other"] += 1
            break

    critical_prefixes = (
        "backbone.vision_backbone.",
        "maskmem_backbone.",
        "transformer.",
        "sam_prompt_encoder.",
        "sam_mask_decoder.",
        "segmentation_head.",
    )
    missing_critical_prefixes = [
        prefix
        for prefix in critical_prefixes
        if any(key.startswith(prefix) for key in model_keys)
        and not any(key.startswith(prefix) for key in covered_targets)
    ]
    for key in (
        "maskmem_tpos_enc",
        "interactivity_no_mem_embed",
        "no_obj_embed_spatial",
        "output_valid_embed",
        "output_invalid_embed",
    ):
        if key in model_keys and key not in covered_targets:
            missing_critical_prefixes.append(key)

    return {
        "checkpoint_keys": len(checkpoint_items),
        "model_keys": len(model_keys),
        "transform_coverage": transform_rows,
        "priority_union": {
            "covered_checkpoint_keys": len(covered_sources),
            "covered_model_keys": len(covered_targets),
            "uncovered_model_keys": max(0, len(model_keys) - len(covered_targets)),
            "source_prefix_counts": source_prefix_counts.most_common(),
            "missing_critical_prefixes": missing_critical_prefixes,
            "covered_model_key_examples": list(covered_targets.keys())[:40],
            "uncovered_model_key_examples": [key for key in sorted(model_keys) if key not in covered_targets][:40],
        },
    }


def _checkpoint_key_summary(checkpoint: Path | None, model: Any | None = None) -> dict[str, Any]:
    if checkpoint is None:
        return {"checkpoint": None, "exists": False}
    path = Path(checkpoint).expanduser()
    summary: dict[str, Any] = {
        "checkpoint": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    if not path.exists():
        return summary
    try:
        import torch

        payload = torch.load(str(path), map_location="cpu", weights_only=True)
        summary["payload_type"] = type(payload).__module__ + "." + type(payload).__name__
        if isinstance(payload, Mapping):
            summary["top_level_keys"] = [str(key) for key in list(payload.keys())[:80]]
            state = payload.get("model") if isinstance(payload.get("model"), Mapping) else payload
            summary["uses_model_key"] = state is not payload
        else:
            state = None
            summary["uses_model_key"] = False
        if isinstance(state, Mapping):
            keys = [str(key) for key in state.keys()]
            summary.update(_key_prefix_summary(keys))
            for prefix in (
                "tracker.",
                "sam2_predictor.",
                "detector.",
                "backbone.",
                "maskmem_backbone.",
                "sam_prompt_encoder.",
                "sam_mask_decoder.",
                "transformer.",
            ):
                summary[f"has_prefix:{prefix}"] = any(key.startswith(prefix) for key in keys)
            for key in (
                "maskmem_tpos_enc",
                "interactivity_no_mem_embed",
                "no_obj_embed_spatial",
                "output_valid_embed",
                "output_invalid_embed",
            ):
                summary[f"has_key:{key}"] = key in state
            if model is not None:
                summary["model_coverage"] = _checkpoint_model_coverage(state, model)
        else:
            summary["state_key_count"] = None
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _load_checkpoint_state(checkpoint: Path) -> Mapping[str, Any]:
    import torch

    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        payload = payload["model"]
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Checkpoint payload is not a state dict: {type(payload).__name__}")
    return payload


def _remap_full_checkpoint_state(state: Mapping[str, Any]) -> dict[str, Any]:
    if not any(str(key).startswith(("sam3_model.", "sam2_predictor.")) for key in state):
        return {str(key): value for key, value in state.items()}
    remapped: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if key.startswith("sam3_model."):
            key = "detector." + key[len("sam3_model.") :]
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key[len("sam2_predictor.") :]
        remapped[key] = value
    return remapped


def _checkpoint_full_model_audit(checkpoint: Path | None, predictor: Any) -> dict[str, Any]:
    """Audit the released merged checkpoint against the final full predictor model."""

    if checkpoint is None:
        return {
            "status": "skipped",
            "passed": False,
            "reason": "no local checkpoint path",
            "blocking_reasons": ["no_local_checkpoint"],
        }
    model = getattr(predictor, "model", predictor)
    try:
        state = _remap_full_checkpoint_state(_load_checkpoint_state(Path(checkpoint)))
        model_state = model.state_dict()
    except Exception as exc:
        return {
            "status": "failed",
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "blocking_reasons": ["checkpoint_audit_failed"],
        }

    checkpoint_keys = set(state)
    model_keys = set(str(key) for key in model_state)
    parameter_keys = set(str(key) for key in dict(model.named_parameters()))
    buffer_keys = set(str(key) for key in dict(model.named_buffers()))
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_bad = sorted(
        key
        for key in checkpoint_keys & model_keys
        if _shape_tuple(state[key]) != _shape_tuple(model_state[key])
    )
    missing_parameters = sorted(set(missing) & parameter_keys)
    missing_buffers = sorted(set(missing) & buffer_keys)
    blocking_reasons: list[str] = []
    if shape_bad:
        blocking_reasons.append("shape_mismatch")
    if missing_parameters:
        blocking_reasons.append("missing_learned_parameters")
    if unexpected:
        blocking_reasons.append("unexpected_checkpoint_tensors")
    return {
        "status": "done",
        "passed": not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(model_keys),
        "missing_total": len(missing),
        "missing_learned_parameters": len(missing_parameters),
        "missing_buffers": len(missing_buffers),
        "unexpected_total": len(unexpected),
        "shape_mismatch_total": len(shape_bad),
        "missing_examples": missing[:40],
        "missing_learned_parameter_examples": missing_parameters[:40],
        "missing_buffer_examples": missing_buffers[:40],
        "unexpected_examples": unexpected[:40],
        "shape_mismatch_examples": shape_bad[:40],
    }


def _build_full_predictor(args: argparse.Namespace, checkpoint: Path | None) -> tuple[Any, dict[str, Any]]:
    import importlib

    builder_mod = importlib.import_module("sam3.model_builder")
    common = {
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "max_num_objects": max(16, int(args.multiplex_count)),
        "multiplex_count": max(1, int(args.multiplex_count)),
        "use_fa3": bool(args.use_fa3),
        "use_rope_real": bool(args.use_rope_real),
        "compile": bool(args.compile),
        "warm_up": False,
        "default_output_prob_thresh": 0.5,
        "async_loading_frames": False,
    }
    if hasattr(builder_mod, "build_sam3_multiplex_video_predictor"):
        builder = getattr(builder_mod, "build_sam3_multiplex_video_predictor")
        return _call_supported(builder, **common), {**common, "builder": "build_sam3_multiplex_video_predictor"}
    if hasattr(builder_mod, "build_sam3_predictor"):
        builder = getattr(builder_mod, "build_sam3_predictor")
        return _call_supported(builder, version="sam3.1", **common), {
            **common,
            "builder": "build_sam3_predictor",
            "version": "sam3.1",
        }
    raise RuntimeError("SAM3 model_builder exposes neither build_sam3_multiplex_video_predictor nor build_sam3_predictor")


def _mask_center_and_box(mask: np.ndarray) -> tuple[list[float], list[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Cannot create a public-session point prompt from an empty object mask")
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    best = int(np.argmin((xs.astype(np.float32) - center_x) ** 2 + (ys.astype(np.float32) - center_y) ** 2))
    return [float(xs[best]), float(ys[best])], [x0, y0, x1 - x0, y1 - y0]


def _probe_public_session(
    args: argparse.Namespace,
    predictor: Any,
    frame_dir: Path,
    annotation: np.ndarray,
    object_ids: list[int],
    target_frame_count: int,
) -> dict[str, Any]:
    """Best-effort check of the official interactive session API.

    This is intentionally diagnostic-only. It checks that explicit object IDs
    can survive a point-prompt session, not that SAM3.1 can consume full masks.
    """

    try:
        import torch

        start = predictor.handle_request(
            request={"type": "start_session", "resource_path": str(frame_dir)}
        )
        session_id = start["session_id"]
        height, width = annotation.shape[:2]
        add_prompt_rows: list[dict[str, Any]] = []
        for object_id in object_ids:
            point_abs, box_abs = _mask_center_and_box(annotation == int(object_id))
            point_rel = [[point_abs[0] / width, point_abs[1] / height]]
            box_rel = [[box_abs[0] / width, box_abs[1] / height, box_abs[2] / width, box_abs[3] / height]]
            add = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": torch.tensor(point_rel, dtype=torch.float32),
                    "point_labels": torch.tensor([1], dtype=torch.int32),
                    "obj_id": int(object_id),
                }
            )
            add_prompt_rows.append(
                {
                    "object_id": int(object_id),
                    "point_abs": point_abs,
                    "point_rel": point_rel[0],
                    "box_abs_reference": box_abs,
                    "box_rel_reference": box_rel[0],
                    "add_prompt_output_keys": sorted(add.get("outputs", {}).keys())
                    if isinstance(add.get("outputs"), Mapping)
                    else [],
                }
            )
        stream_rows: list[dict[str, Any]] = []
        for response in predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
            }
        ):
            row = {
                "frame_index": int(response.get("frame_index", response.get("frame_idx", len(stream_rows)))),
                "output_keys": sorted(response.get("outputs", {}).keys()) if isinstance(response.get("outputs"), Mapping) else [],
            }
            try:
                frame_idx, output_ids, logits, _scores = _normalize_propagation_item(response)
                row.update(
                    {
                        "normalized_frame_index": int(frame_idx),
                        "output_object_ids": [int(value) for value in output_ids],
                        "mask_shape": list(logits.shape),
                    }
                )
            except Exception as exc:
                row["normalize_error"] = f"{type(exc).__name__}: {exc}"
            stream_rows.append(row)
            if len(stream_rows) >= target_frame_count:
                break
        try:
            predictor.handle_request(request={"type": "reset_session", "session_id": session_id})
        except Exception:
            pass
        expected = set(int(object_id) for object_id in object_ids)
        rows_with_ids = [
            set(int(value) for value in row.get("output_object_ids", []))
            for row in stream_rows
            if "output_object_ids" in row
        ]
        return {
            "status": "done",
            "prompt": "per_object_positive_point_from_first_frame_mask",
            "candidate_mode": "point_prompt_only",
            "mask_prompt_used": False,
            "expected_object_ids": sorted(expected),
            "added_prompts": add_prompt_rows,
            "maintained_expected_object_ids": bool(rows_with_ids)
            and all(expected.issubset(ids) for ids in rows_with_ids),
            "propagation": stream_rows,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _prepare_probe_inputs(args: argparse.Namespace, exp_dir: Path, video: Any) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    cache_dir = exp_dir / "cache"
    prepared_frames = _prepare_frame_dir(
        video,
        {
            "data_root": str(data_root),
            "cache_dir": str(cache_dir),
            "resize_long_side": 0,
        },
    )
    target_frame_count = min(len(prepared_frames), max(1, int(args.max_frames)))
    first = prepared_frames[0]
    annotation = _load_mask_prompt(video.prompts, data_root)
    if annotation.ndim == 3:
        annotation = np.any(annotation[..., :3] > 0, axis=-1).astype(np.uint8)
    annotation = _resize_mask(annotation, first.original_width, first.original_height)
    object_masks = _object_masks_from_initial_mask(annotation)
    object_ids = sorted(int(object_id) for object_id in object_masks)
    if not object_ids:
        raise RuntimeError(f"{video.video_id}: first-frame annotation contains no object IDs")
    frame_dir = _session_frame_dir(prepared_frames, target_frame_count, cache_dir, video.video_id)
    return {
        "data_root": data_root,
        "cache_dir": cache_dir,
        "prepared_frames": prepared_frames,
        "target_frame_count": target_frame_count,
        "annotation": annotation,
        "object_masks": object_masks,
        "object_ids": object_ids,
        "frame_dir": frame_dir,
    }


def _probe_state(args: argparse.Namespace, exp_dir: Path, model: Any, video: Any, checkpoint: Path | None) -> dict[str, Any]:
    import torch

    inputs = _prepare_probe_inputs(args, exp_dir, video)
    target_frame_count = int(inputs["target_frame_count"])
    annotation = inputs["annotation"]
    object_masks = inputs["object_masks"]
    object_ids = inputs["object_ids"]
    frame_dir = inputs["frame_dir"]
    state = _initialize_native_state(
        model,
        frame_dir,
        offload_video=bool(args.offload_video_to_cpu),
        offload_state=bool(args.offload_state_to_cpu),
    )
    snapshots = [_state_snapshot(state, object_ids, model, "after_init_state")]

    mask_tensor = torch.from_numpy(np.stack([object_masks[object_id] for object_id in object_ids])).bool()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.no_grad()
    propagation_rows: list[dict[str, Any]] = []
    conditioning_info: dict[str, Any] = {}
    with torch.inference_mode(), autocast:
        model.add_new_masks(
            inference_state=state,
            frame_idx=0,
            obj_ids=object_ids,
            masks=mask_tensor,
            add_mask_to_memory=True,
        )
        conditioning_info = {
            "mask_conditioning": "official_add_new_masks",
            "num_conditioned_objects": len(object_ids),
            "conditioned_object_ids": [int(object_id) for object_id in object_ids],
            "cached_conditioning_frames": [0],
            "private_state_used": False,
        }
        snapshots.append(_state_snapshot(state, object_ids, model, "after_mask_conditioning"))
        propagation = _call_supported(
            model.propagate_in_video,
            inference_state=state,
            start_frame_idx=0,
            max_frame_num_to_track=target_frame_count - 1,
            reverse=False,
            tqdm_disable=True,
            run_mem_encoder=True,
        )
        for item in propagation:
            frame_idx, output_ids, logits, object_scores = _normalize_propagation_item(item)
            if frame_idx < 0 or frame_idx >= target_frame_count:
                continue
            raw_output_ids = [int(object_id) for object_id in output_ids]
            reordered_logits = _reorder_objects(logits, raw_output_ids, object_ids, allow_missing=True)
            reordered_scores = _reorder_scores(object_scores, raw_output_ids, object_ids, allow_missing=True)
            object_rows: list[dict[str, Any]] = []
            for index, object_id in enumerate(object_ids):
                binary = reordered_logits[index] > 0
                score_logit = (
                    float(reordered_scores[index])
                    if reordered_scores is not None and index < len(reordered_scores)
                    else None
                )
                object_rows.append(
                    {
                        "object_id": int(object_id),
                        "returned_by_predictor": int(object_id) in set(raw_output_ids),
                        "presence_logit": score_logit,
                        "presence": _sigmoid(score_logit),
                        "foreground_pixels": int(binary.sum()),
                        "foreground_fraction": float(binary.mean()),
                        "mask_logit_min": float(np.min(reordered_logits[index])),
                        "mask_logit_max": float(np.max(reordered_logits[index])),
                    }
                )
            propagation_rows.append(
                {
                    "frame_index": int(frame_idx),
                    "raw_output_object_ids": raw_output_ids,
                    "missing_expected_object_ids": sorted(set(object_ids) - set(raw_output_ids)),
                    "extra_object_ids": sorted(set(raw_output_ids) - set(object_ids)),
                    "internal_tracker_recovery": None,
                    "objects": object_rows,
                    "raw_item_type": type(item).__module__ + "." + type(item).__name__,
                }
            )
            snapshots.append(_state_snapshot(state, object_ids, model, f"after_propagation_frame_{frame_idx}"))
            if len(propagation_rows) >= target_frame_count:
                break

    reset = getattr(model, "reset_state", None)
    if callable(reset):
        try:
            reset(state)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "video_id": video.video_id,
        "target_frame_count": target_frame_count,
        "frame_dir": str(frame_dir),
        "object_ids": object_ids,
        "initial_mask_shape": list(annotation.shape),
        "initial_object_pixel_counts": {
            str(object_id): int(object_masks[object_id].sum())
            for object_id in object_ids
        },
        "conditioning_info": conditioning_info,
        "propagation": propagation_rows,
        "snapshots": snapshots,
        "checkpoint": str(checkpoint) if checkpoint else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    exp_dir = Path(args.output_dir).expanduser().resolve() / args.experiment_id
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.sam3_repo_dir:
        repo_dir = Path(args.sam3_repo_dir).expanduser().resolve()
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
    if args.install_sam3:
        install_dir = args.sam3_repo_dir or str(exp_dir / "external" / "sam3")
        install_status = install_sam3_if_requested(install_dir, requested=True)
        _atomic_json(logs_dir / "sam31_install_status.json", install_status.to_dict())
        if not install_status.available:
            print(json.dumps({"status": "failed_install", "error": install_status.reason}, indent=2))
            return 1

    try:
        checkpoint = _resolve_checkpoint(args, exp_dir)
        try:
            predictor, build_config = _build_full_predictor(args, checkpoint)
        except Exception as exc:
            payload = {
                "status": "failed_full_predictor_build",
                "checkpoint": str(checkpoint),
                "sam3_checkpoint_load_target": "full_multiplex_predictor",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            _atomic_json(logs_dir / "sam31_api_introspection.json", payload)
            print(json.dumps(payload, indent=2, ensure_ascii=True))
            return 1

        model = getattr(predictor, "model", predictor)
        audit = _checkpoint_full_model_audit(checkpoint, predictor)
        api_payload = {
            "status": "built",
            "experiment_id": args.experiment_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "sam3_checkpoint_load_target": "full_multiplex_predictor",
            "sam3_submission_status": "blocked_public_mask_session_api",
            "sam3_public_mask_request": False,
            "sam3_low_level_add_new_masks": True,
            "sam3_low_level_standalone_video_api": False,
            "sam3_candidate_mode": "point_prompt_only",
            "private_state_used": False,
            "previous_mask_recovery": False,
            "checkpoint_keys": _checkpoint_key_summary(checkpoint, model),
            "checkpoint_full_model_audit": audit,
            "model_state_keys": _model_state_key_summary(model),
            "build": {
                "available": True,
                "build_config": build_config,
            },
            "predictor_class": f"{type(predictor).__module__}.{type(predictor).__name__}",
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "runtime": _runtime_version(model),
            "predictor_callables": _callable_inventory(predictor),
            "model_callables": _callable_inventory(model),
        }
        _atomic_json(logs_dir / "sam31_api_introspection.json", api_payload)
        if not bool(audit.get("passed")):
            payload = {
                "status": "failed_checkpoint_audit",
                "checkpoint": str(checkpoint),
                "api_introspection": str(logs_dir / "sam31_api_introspection.json"),
                "blocking_reasons": audit.get("blocking_reasons", []),
                "sam3_submission_status": "blocked_public_mask_session_api",
            }
            _atomic_json(logs_dir / "summary.json", payload)
            print(json.dumps(payload, indent=2, ensure_ascii=True))
            return 1

        video = _select_video(Path(args.data_root).expanduser().resolve(), args.video_id)
        inputs = _prepare_probe_inputs(args, exp_dir, video)
        public_probe_payload = {
            "video_id": video.video_id,
            "target_frame_count": int(inputs["target_frame_count"]),
            "frame_dir": str(inputs["frame_dir"]),
            "object_ids": inputs["object_ids"],
            "initial_mask_shape": list(inputs["annotation"].shape),
            "initial_object_pixel_counts": {
                str(object_id): int(inputs["object_masks"][object_id].sum())
                for object_id in inputs["object_ids"]
            },
            "checkpoint": str(checkpoint) if checkpoint else None,
            "public_session_probe": _probe_public_session(
                args,
                predictor,
                inputs["frame_dir"],
                inputs["annotation"],
                inputs["object_ids"],
                int(inputs["target_frame_count"]),
            ),
        }
        _atomic_json(logs_dir / "sam31_public_session_probe.json", public_probe_payload)

        low_level_probe_payload: dict[str, Any] | None = None
        if args.run_low_level_mask_probe:
            low_level_build = build_sam3_tracker(
                checkpoint_path=checkpoint,
                device="cuda",
                multiplex_count=args.multiplex_count,
                use_fa3=args.use_fa3,
                use_rope_real=args.use_rope_real,
                compile_model=args.compile,
                strict_runtime=not args.allow_unsupported_runtime,
                run_mode=SAM3_RUN_MODE_OFFICIAL,
            )
            if not low_level_build.available or low_level_build.predictor is None:
                low_level_probe_payload = {
                    "status": "failed_low_level_build",
                    "checkpoint": str(checkpoint),
                    "build": low_level_build.to_dict(),
                }
            else:
                low_level_probe_payload = _probe_state(args, exp_dir, low_level_build.predictor, video, checkpoint)
            _atomic_json(logs_dir / "sam31_low_level_mask_probe.json", low_level_probe_payload)

        public_session = public_probe_payload.get("public_session_probe", {})
        public_session_ok = (
            public_session.get("status") == "done"
            and bool(public_session.get("maintained_expected_object_ids"))
        )
        summary = {
            "status": "done" if public_session_ok else "failed_public_session_probe",
            "experiment_dir": str(exp_dir),
            "api_introspection": str(logs_dir / "sam31_api_introspection.json"),
            "public_session_probe": str(logs_dir / "sam31_public_session_probe.json"),
            "low_level_mask_probe": str(logs_dir / "sam31_low_level_mask_probe.json") if low_level_probe_payload is not None else None,
            "video_id": args.video_id,
            "object_ids": public_probe_payload["object_ids"],
            "sam3_checkpoint_load_target": "full_multiplex_predictor",
            "sam3_submission_status": "blocked_public_mask_session_api",
            "sam3_candidate_mode": "point_prompt_only",
            "sam3_public_mask_request": False,
            "sam3_low_level_add_new_masks": True,
            "sam3_low_level_standalone_video_api": False,
            "mask_api_path": None,
            "private_state_used": False,
            "previous_mask_recovery": False,
            "checkpoint_audit_passed": bool(audit.get("passed")),
            "public_session_probe_status": public_session.get("status"),
            "public_session_maintained_expected_object_ids": public_session.get("maintained_expected_object_ids"),
            "propagation": public_session.get("propagation", []),
        }
        _atomic_json(logs_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
        return 0 if public_session_ok else 1
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _atomic_json(logs_dir / "summary.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

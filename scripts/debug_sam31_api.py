"""Probe SAM 3.1 full-predictor mask conditioning state.

This script is intentionally diagnostic-only. It builds the official SAM 3.1
full video predictor, conditions one short real SUFE video with the complete
first-frame mask, and writes compact JSON snapshots of the predictor API,
state keys, tracker metadata, action history, and first propagation outputs.
It does not create masks or a submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import os
import sys
import traceback
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
    SAM3_RUN_MODE_FULL,
    _call_supported,
    _full_predictor_model,
    _init_full_predictor_mask_state,
    _initialize_native_state,
    _normalize_propagation_item,
    _reorder_objects,
    _reorder_scores,
    _runtime_version,
    _session_frame_dir,
    _set_full_predictor_tracking_bounds,
    _sigmoid,
    _state_summary,
    _to_numpy,
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
    parser = argparse.ArgumentParser(description="Debug SAM 3.1 full predictor mask-conditioning state.")
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


def _probe_state(args: argparse.Namespace, exp_dir: Path, model: Any, video: Any) -> dict[str, Any]:
    import torch

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
    state = _initialize_native_state(
        model,
        frame_dir,
        offload_video=bool(args.offload_video_to_cpu),
        offload_state=bool(args.offload_state_to_cpu),
    )
    snapshots = [_state_snapshot(state, object_ids, model, "after_init_state")]

    mask_tensor = torch.from_numpy(np.stack([object_masks[object_id] for object_id in object_ids])).float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else torch.no_grad()
    propagation_rows: list[dict[str, Any]] = []
    conditioning_info: dict[str, Any] = {}
    with torch.inference_mode(), autocast:
        _set_full_predictor_tracking_bounds(state, 0, target_frame_count - 1)
        snapshots.append(_state_snapshot(state, object_ids, model, "after_tracking_bounds_before_conditioning"))
        conditioning_info = _init_full_predictor_mask_state(model, state, 0, object_ids, mask_tensor)
        snapshots.append(_state_snapshot(state, object_ids, model, "after_mask_conditioning"))
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
        build = build_sam3_tracker(
            checkpoint_path=checkpoint,
            device="cuda",
            multiplex_count=args.multiplex_count,
            use_fa3=args.use_fa3,
            use_rope_real=args.use_rope_real,
            compile_model=args.compile,
            strict_runtime=not args.allow_unsupported_runtime,
            run_mode=SAM3_RUN_MODE_FULL,
        )
        if not build.available or build.predictor is None:
            payload = {"status": "failed_build", "checkpoint": str(checkpoint), "build": build.to_dict()}
            _atomic_json(logs_dir / "sam31_api_introspection.json", payload)
            print(json.dumps(payload, indent=2, ensure_ascii=True))
            return 1

        predictor = build.predictor
        model = _full_predictor_model(predictor)
        api_payload = {
            "status": "built",
            "experiment_id": args.experiment_id,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "build": build.to_dict(),
            "predictor_class": f"{type(predictor).__module__}.{type(predictor).__name__}",
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "runtime": _runtime_version(model),
            "predictor_callables": _callable_inventory(predictor),
            "model_callables": _callable_inventory(model),
        }
        _atomic_json(logs_dir / "sam31_api_introspection.json", api_payload)

        video = _select_video(Path(args.data_root).expanduser().resolve(), args.video_id)
        probe_payload = _probe_state(args, exp_dir, model, video)
        _atomic_json(logs_dir / "sam31_state_probe.json", probe_payload)

        summary = {
            "status": "done",
            "experiment_dir": str(exp_dir),
            "api_introspection": str(logs_dir / "sam31_api_introspection.json"),
            "state_probe": str(logs_dir / "sam31_state_probe.json"),
            "video_id": args.video_id,
            "object_ids": probe_payload["object_ids"],
            "propagation": [
                {
                    "frame_index": row["frame_index"],
                    "raw_output_object_ids": row["raw_output_object_ids"],
                    "missing_expected_object_ids": row["missing_expected_object_ids"],
                }
                for row in probe_payload["propagation"]
            ],
        }
        _atomic_json(logs_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
        return 0
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

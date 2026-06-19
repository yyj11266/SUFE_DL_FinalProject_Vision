from __future__ import annotations

import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from src.trackers.sam3_tracker_optional import (
    SAM3_RUN_MODE_OFFICIAL,
    SAM3_RUN_MODE_FULL,
    SAM3_RUN_MODE_LOW_LEVEL,
    Sam3Availability,
    _patch_native_tracker_forward_image_for_mask_tracking,
    _remap_full_multiplex_checkpoint_for_native_tracker,
    build_sam3_tracker,
    run_sam3_video_with_mask_prompt,
)


class FakeNativeSam31:
    def __init__(self, drop_object_after_frame: int | None = None, empty_after_frame: int | None = None) -> None:
        self.drop_object_after_frame = drop_object_after_frame
        self.empty_after_frame = empty_after_frame
        self.add_new_masks_calls = 0
        self.added_object_ids: list[int] = []

    def init_state(self, video_path: str, **_: object) -> dict[str, object]:
        frame_count = len(list(Path(video_path).glob("*.jpg")))
        return {
            "num_frames": frame_count,
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "frames_already_tracked": {},
            "output_dict_per_obj": {},
            "first_ann_frame_idx": 0,
        }

    def add_new_masks(self, inference_state: dict[str, object], frame_idx: int, obj_ids: list[int], masks: torch.Tensor, **_: object) -> None:
        self.add_new_masks_calls += 1
        self.added_object_ids = [int(object_id) for object_id in obj_ids]
        inference_state["obj_ids"] = list(obj_ids)
        inference_state["initial_masks"] = masks.float()

    def propagate_in_video(self, inference_state: dict[str, object], max_frame_num_to_track: int, **_: object):
        object_ids = inference_state["obj_ids"]
        initial = inference_state["initial_masks"]
        assert isinstance(object_ids, list)
        assert isinstance(initial, torch.Tensor)
        output_dict = inference_state["output_dict"]
        assert isinstance(output_dict, dict)
        for frame_index in range(max_frame_num_to_track + 1):
            frame_object_ids = list(object_ids)
            if self.drop_object_after_frame is not None and frame_index >= self.drop_object_after_frame and len(frame_object_ids) > 1:
                frame_object_ids = frame_object_ids[:-1]
            mask_indices = [object_ids.index(object_id) for object_id in frame_object_ids]
            frame_masks = initial[mask_indices] if mask_indices else initial[:0]
            if self.empty_after_frame is not None and frame_index >= self.empty_after_frame:
                frame_masks = torch.zeros_like(frame_masks)
            logits = torch.where(frame_masks > 0, torch.tensor(8.0), torch.tensor(-8.0)).unsqueeze(1)
            scores = torch.full((len(frame_object_ids), 1), 4.0)
            storage = "cond_frame_outputs" if frame_index == 0 else "non_cond_frame_outputs"
            output_dict[storage][frame_index] = {
                "pred_masks": logits,
                "object_score_logits": scores,
                "iou_score": torch.full((len(object_ids), 1), 0.9),
            }
            yield frame_index, frame_object_ids, None, logits, scores


class FakeFullSam31Model:
    def __init__(
        self,
        drop_object_after_frame: int | None = None,
        empty_internal_after_frame: int | None = None,
        dominate_first_object_after_frame: int | None = None,
    ) -> None:
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.is_multiplex = True
        self.masklet_confirmation_enable = True
        self.masklet_confirmation_consecutive_det_thresh = 2
        self.conditioned_masks: torch.Tensor | None = None
        self.conditioned_ids: list[int] = []
        self.cache_calls: list[int] = []
        self.drop_object_after_frame = drop_object_after_frame
        self.empty_internal_after_frame = empty_internal_after_frame
        self.dominate_first_object_after_frame = dominate_first_object_after_frame

    def init_state(self, resource_path: str, **_: object) -> dict[str, object]:
        frame_count = len(list(Path(resource_path).glob("*.jpg")))
        return {
            "num_frames": frame_count,
            "orig_height": 12,
            "orig_width": 16,
            "device": self.device,
            "sam2_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "cached_frame_outputs": {},
            "action_history": [],
        }

    def _prepare_backbone_feats(self, inference_state: dict[str, object], frame_idx: int, reverse: bool) -> None:
        feature_cache = inference_state.setdefault("feature_cache", {})
        assert isinstance(feature_cache, dict)
        feature_cache["prepared"] = {"frame_idx": frame_idx, "reverse": reverse}

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: list[int],
        new_obj_masks: torch.Tensor,
        tracker_states_local: list[dict[str, object]],
        **_: object,
    ) -> list[dict[str, object]]:
        self.conditioned_ids = list(new_obj_ids)
        self.conditioned_masks = new_obj_masks.detach().float().cpu()
        self.assert_mask_like(new_obj_masks)
        tracker_states_local.append(
            {
                "obj_ids": list(new_obj_ids),
                "frame_idx": frame_idx,
                "num_frames": num_frames,
                "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                "output_dict_per_obj": {},
            }
        )
        return tracker_states_local

    @staticmethod
    def assert_mask_like(masks: torch.Tensor) -> None:
        assert masks.ndim == 3
        assert masks.is_floating_point()

    def _initialize_metadata(self) -> dict[str, object]:
        return {
            "obj_ids_per_gpu": [np.array([], dtype=np.int64)],
            "obj_ids_all_gpu": np.array([], dtype=np.int64),
            "num_obj_per_gpu": np.zeros(1, dtype=np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_sam2_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
            "num_buc_per_gpu": np.zeros(1, dtype=np.int64),
            "rank0_metadata": {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(int),
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(set),
                "masklet_confirmation": {"status": np.array([], dtype=np.int64), "consecutive_det_num": np.array([], dtype=np.int64)},
            },
            "gpu_metadata": {"N_obj": 0},
        }

    def _cache_frame_outputs(self, inference_state: dict[str, object], frame_idx: int, obj_id_to_mask: dict[int, torch.Tensor]) -> None:
        self.cache_calls.append(int(frame_idx))
        inference_state["cached_frame_outputs"][frame_idx] = dict(obj_id_to_mask)

    def _count_buckets_in_states(self, states: list[dict[str, object]]) -> int:
        return len(states)

    def add_action_history(self, inference_state: dict[str, object], action_type: str, frame_idx: int | None = None, obj_ids: list[int] | None = None) -> None:
        inference_state["action_history"].append({"type": action_type, "frame_idx": frame_idx, "obj_ids": obj_ids})

    def propagate_in_video(self, inference_state: dict[str, object], start_frame_idx: int, max_frame_num_to_track: int, **_: object):
        assert inference_state["action_history"][0]["type"] == "add"
        feature_cache = inference_state["feature_cache"]
        assert isinstance(feature_cache, dict)
        assert feature_cache["tracking_bounds"] == {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }
        assert self.conditioned_masks is not None
        object_ids = list(self.conditioned_ids)
        masks = self.conditioned_masks > 0
        sam2_state = inference_state["sam2_inference_states"][0]
        assert isinstance(sam2_state, dict)
        output_dict = sam2_state["output_dict"]
        assert isinstance(output_dict, dict)
        for frame_index in range(start_frame_idx, max_frame_num_to_track + 1):
            internal_masks = masks
            if self.empty_internal_after_frame is not None and frame_index >= self.empty_internal_after_frame:
                internal_masks = torch.zeros_like(masks, dtype=torch.bool)
            internal_logits = torch.where(internal_masks, torch.tensor(8.0), torch.tensor(-8.0)).unsqueeze(1)
            if (
                self.dominate_first_object_after_frame is not None
                and frame_index >= self.dominate_first_object_after_frame
                and len(object_ids) > 1
            ):
                internal_logits[0, 0] = torch.where(masks[1], torch.tensor(1.0), torch.tensor(-8.0))
            storage_key = "cond_frame_outputs" if frame_index == 0 else "non_cond_frame_outputs"
            output_dict[storage_key][frame_index] = {
                "pred_masks": internal_logits,
                "object_score_logits": torch.full((len(object_ids), 1), 4.0),
                "local_obj_id_to_idx": {int(object_id): index for index, object_id in enumerate(object_ids)},
            }
            frame_object_ids = list(object_ids)
            if self.drop_object_after_frame is not None and frame_index >= self.drop_object_after_frame and len(frame_object_ids) > 1:
                frame_object_ids = frame_object_ids[:-1]
            output_ids = np.asarray(list(reversed(frame_object_ids)), dtype=np.int64)
            reordered_masks = torch.stack([masks[object_ids.index(int(object_id))] for object_id in output_ids], dim=0).numpy()
            yield frame_index, {
                "out_obj_ids": output_ids,
                "out_binary_masks": reordered_masks,
                "out_probs": np.full(len(output_ids), 0.95, dtype=np.float32),
            }

    def reset_state(self, inference_state: dict[str, object]) -> None:
        inference_state["sam2_inference_states"] = []


class FakeFullSam31Predictor:
    def __init__(self, model: FakeFullSam31Model | None = None) -> None:
        self.model = model or FakeFullSam31Model()


class MissingMaskApi:
    def init_state(self, *_: object, **__: object) -> dict[str, object]:
        return {}

    def propagate_in_video(self, *_: object, **__: object):
        return iter(())


class TinyTrackerState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("maskmem_tpos_enc", torch.nn.Parameter(torch.zeros(1)))
        self.backbone = torch.nn.Module()
        self.backbone.vision_backbone = torch.nn.Module()
        self.backbone.vision_backbone.register_parameter("trunk_weight", torch.nn.Parameter(torch.zeros(2, 2)))
        self.maskmem_backbone = torch.nn.Linear(2, 2)
        self.transformer = torch.nn.Linear(2, 2)
        self.sam_prompt_encoder = torch.nn.Linear(2, 2)
        self.sam_mask_decoder = torch.nn.Linear(2, 2)


class ForwardImageRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def forward_image(self, *args: object, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {"ok": True}


class Sam31AdapterTest(unittest.TestCase):
    def _video(self, root: Path) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
        frame_dir = root / "JPEGImages" / "v1"
        mask_dir = root / "Annotations" / "v1"
        frame_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        frames = []
        for index in range(3):
            path = frame_dir / f"{index:05d}.jpg"
            Image.fromarray(np.full((12, 16, 3), 80 + index, dtype=np.uint8)).save(path)
            frames.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "frame_stem": f"{index:05d}",
                }
            )
        annotation = np.zeros((12, 16), dtype=np.uint8)
        annotation[1:5, 1:5] = 1
        annotation[6:10, 9:14] = 2
        mask_path = mask_dir / "00000.png"
        Image.fromarray(annotation).save(mask_path)
        video = {"video_id": "v1", "frames": frames, "relative_path": "JPEGImages/v1"}
        prompts = [{"prompt_type": "mask", "relative_path": mask_path.relative_to(root).as_posix()}]
        return video, prompts, annotation

    def test_official_mask_api_path_and_first_frame_exactness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, annotation = self._video(root)
            predictor = FakeNativeSam31()
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_OFFICIAL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "save_native_scores": True,
                },
            )
            self.assertEqual(result.status, "done", result.error)
            self.assertEqual(result.object_ids, [1, 2])
            self.assertEqual(predictor.add_new_masks_calls, 1)
            self.assertEqual(predictor.added_object_ids, [1, 2])
            self.assertTrue(result.first_frame_exact)
            self.assertTrue(np.array_equal(np.asarray(Image.open(result.mask_paths[0])), annotation))
            self.assertTrue(Path(result.native_scores_path or "").exists())
            self.assertFalse(result.fallback_used)
            self.assertFalse(result.diagnostics["private_state_used"])
            self.assertEqual(result.diagnostics["sam3_official_api_path"], "add_new_masks")

    def test_official_mask_api_missing_object_id_is_diagnosed_without_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, _ = self._video(root)
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": FakeNativeSam31(drop_object_after_frame=1),
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_OFFICIAL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "save_native_scores": True,
                },
            )
            self.assertEqual(result.status, "done", result.error)
            self.assertFalse(result.fallback_used)
            self.assertEqual(result.diagnostics["internal_tracker_recovery_events"], [])
            self.assertGreater(result.diagnostics["total_missing_output_frames"], 0)
            self.assertEqual(result.diagnostics["per_object"]["2"]["first_missing_output_frame"], 1)
            self.assertTrue(any("mask API omitted expected object IDs" in warning for warning in result.warnings))

    def test_official_mask_api_rejects_previous_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, _ = self._video(root)
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": FakeNativeSam31(),
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_OFFICIAL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "sam3_empty_mask_policy": "previous",
                },
            )
            self.assertEqual(result.status, "failed")
            self.assertIn("official_mask_api does not allow", result.error or "")

    def test_missing_mask_api_fails_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, _ = self._video(root)
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": MissingMaskApi(),
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_OFFICIAL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                },
            )
            self.assertEqual(result.status, "failed")
            self.assertIn("add_new_masks", result.error or "")
            self.assertFalse(result.fallback_used)

    def test_full_predictor_mask_path_and_high_level_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, annotation = self._video(root)
            predictor = FakeFullSam31Predictor()
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_FULL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "save_native_scores": True,
                },
            )

            self.assertEqual(result.status, "done", result.error)
            self.assertEqual(result.object_ids, [1, 2])
            self.assertEqual(predictor.model.conditioned_ids, [1, 2])
            self.assertIsNotNone(predictor.model.conditioned_masks)
            self.assertTrue(result.first_frame_exact)
            self.assertTrue(np.array_equal(np.asarray(Image.open(result.mask_paths[0])), annotation))
            self.assertTrue(Path(result.native_scores_path or "").exists())
            self.assertEqual(predictor.model.cache_calls, [0])

    def test_full_predictor_missing_object_id_is_diagnosed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, _ = self._video(root)
            predictor = FakeFullSam31Predictor(FakeFullSam31Model(drop_object_after_frame=1))
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_FULL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "save_native_scores": True,
                    "sam3_recover_internal_tracker_outputs": False,
                },
            )

            self.assertEqual(result.status, "done", result.error)
            self.assertGreater(result.diagnostics["total_missing_output_frames"], 0)
            self.assertTrue(result.diagnostics["missing_output_events"])
            self.assertTrue(any("omitted expected object IDs" in warning for warning in result.warnings))
            object_two = result.diagnostics["per_object"]["2"]
            self.assertEqual(object_two["first_missing_output_frame"], 1)
            self.assertEqual(object_two["missing_output_frames"], 2)

    def test_full_predictor_recovers_missing_ids_from_internal_tracker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, annotation = self._video(root)
            predictor = FakeFullSam31Predictor(FakeFullSam31Model(drop_object_after_frame=1))
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_FULL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "save_native_scores": True,
                },
            )

            self.assertEqual(result.status, "done", result.error)
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.diagnostics["total_missing_output_frames"], 0)
            self.assertEqual(result.diagnostics["internal_tracker_recovery_frames"], 2)
            self.assertTrue(result.diagnostics["internal_tracker_recovery_events"])
            self.assertTrue(any("Recovered omitted SAM 3.1" in warning for warning in result.warnings))
            self.assertTrue(np.array_equal(np.asarray(Image.open(result.mask_paths[0])), annotation))
            recovered_mask = np.asarray(Image.open(result.mask_paths[1]))
            self.assertEqual(set(np.unique(recovered_mask).tolist()), {0, 1, 2})

    def test_full_predictor_previous_policy_fills_empty_recovered_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, annotation = self._video(root)
            predictor = FakeFullSam31Predictor(
                FakeFullSam31Model(drop_object_after_frame=1, empty_internal_after_frame=1)
            )
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_FULL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "sam3_empty_mask_policy": "previous",
                    "save_native_scores": True,
                },
            )

            self.assertEqual(result.status, "done", result.error)
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.diagnostics["total_missing_output_frames"], 0)
            self.assertEqual(result.diagnostics["empty_mask_policy"], "previous")
            self.assertEqual(result.diagnostics["empty_mask_policy_frames"], 2)
            self.assertEqual(len(result.diagnostics["empty_mask_policy_events"]), 4)
            self.assertEqual(
                {event["object_id"] for event in result.diagnostics["empty_mask_policy_events"]},
                {1, 2},
            )
            self.assertEqual(result.diagnostics["per_object"]["2"]["non_first_zero_frames"], 0)
            self.assertTrue(any("empty-mask policy" in warning for warning in result.warnings))
            self.assertTrue(np.array_equal(np.asarray(Image.open(result.mask_paths[0])), annotation))
            stabilized_mask = np.asarray(Image.open(result.mask_paths[1]))
            self.assertEqual(set(np.unique(stabilized_mask).tolist()), {0, 1, 2})

    def test_full_predictor_indexed_absence_policy_restores_lost_object_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, prompts, _ = self._video(root)
            predictor = FakeFullSam31Predictor(
                FakeFullSam31Model(drop_object_after_frame=1, dominate_first_object_after_frame=1)
            )
            result = run_sam3_video_with_mask_prompt(
                video,
                prompts,
                root / "outputs",
                {
                    "data_root": str(root),
                    "cache_dir": str(root / "cache"),
                    "predictor": predictor,
                    "device": "cpu",
                    "sam3_run_mode": SAM3_RUN_MODE_FULL,
                    "prompt_mode": "mask",
                    "resize_long_side": 0,
                    "output_frame_stems": ["00000", "00001", "00002"],
                    "sam3_indexed_absence_policy": "previous",
                    "save_native_scores": True,
                },
            )

            self.assertEqual(result.status, "done", result.error)
            self.assertTrue(result.fallback_used)
            self.assertEqual(result.diagnostics["indexed_absence_policy"], "previous")
            self.assertEqual(result.diagnostics["indexed_absence_policy_frames"], 2)
            self.assertEqual(len(result.diagnostics["indexed_absence_policy_events"]), 2)
            self.assertTrue(any("indexed-absence policy" in warning for warning in result.warnings))
            stabilized_mask = np.asarray(Image.open(result.mask_paths[1]))
            self.assertEqual(set(np.unique(stabilized_mask).tolist()), {0, 1, 2})

    def test_build_defaults_to_official_mask_api_builder(self) -> None:
        fake_model = FakeNativeSam31()
        fake_builder_mod = mock.Mock()
        fake_builder_mod.build_sam3_multiplex_video_model.return_value = fake_model
        status = Sam3Availability(True, "available")

        with mock.patch("src.trackers.sam3_tracker_optional.check_sam3_available", return_value=status), mock.patch(
            "src.trackers.sam3_tracker_optional.importlib.import_module",
            return_value=fake_builder_mod,
        ):
            result = build_sam3_tracker(checkpoint_path="/tmp/sam3.1_multiplex.pt", device="cpu")

        self.assertTrue(result.available, result.error)
        fake_builder_mod.build_sam3_multiplex_video_model.assert_called_once()
        self.assertEqual(result.build_config["builder"], "build_sam3_multiplex_video_model/add_new_masks")

    def test_sam31_modes_cannot_make_submission(self) -> None:
        from scripts.run_sam31_vos import main

        for run_mode in ("official_mask_api", "low_level_debug", "full_predictor_mask"):
            with self.subTest(run_mode=run_mode), self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--data-root",
                        "/tmp/missing",
                        "--output-dir",
                        "/tmp/out",
                        "--sam3-run-mode",
                        run_mode,
                        "--make-submission",
                    ]
                )
            self.assertIn("SAM3.1 --make-submission is blocked", str(raised.exception))

    def test_official_cli_rejects_previous_policy(self) -> None:
        from scripts.run_sam31_vos import main

        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "--data-root",
                    "/tmp/missing",
                    "--output-dir",
                    "/tmp/out",
                    "--sam3-run-mode",
                    "official_mask_api",
                    "--sam3-empty-mask-policy",
                    "previous",
                ]
            )
        self.assertIn("official_mask_api does not allow", str(raised.exception))

    def test_full_multiplex_checkpoint_key_remap(self) -> None:
        model = TinyTrackerState()
        checkpoint = {
            "tracker.model.maskmem_tpos_enc": torch.ones(1),
            "detector.backbone.vision_backbone.trunk_weight": torch.ones(2, 2),
            "tracker.model.maskmem_backbone.weight": torch.ones(2, 2),
            "tracker.model.maskmem_backbone.bias": torch.ones(2),
            "tracker.model.transformer.weight": torch.ones(2, 2),
            "tracker.model.transformer.bias": torch.ones(2),
            "tracker.model.sam_prompt_encoder.weight": torch.ones(2, 2),
            "tracker.model.sam_prompt_encoder.bias": torch.ones(2),
            "tracker.model.sam_mask_decoder.weight": torch.ones(2, 2),
            "tracker.model.sam_mask_decoder.bias": torch.ones(2),
            "detector.unused.weight": torch.ones(3),
        }

        remapped, diagnostics = _remap_full_multiplex_checkpoint_for_native_tracker(model, checkpoint)

        self.assertIn("maskmem_tpos_enc", remapped)
        self.assertIn("backbone.vision_backbone.trunk_weight", remapped)
        self.assertEqual(diagnostics["missing_critical_prefixes"], [])
        self.assertEqual(diagnostics["source_prefix_counts"]["tracker.model"], 9)
        self.assertEqual(diagnostics["source_prefix_counts"]["detector.backbone.vision_backbone"], 1)

    def test_mask_tracking_forward_image_patch_disables_root_sam3_output(self) -> None:
        model = ForwardImageRecorder()
        _patch_native_tracker_forward_image_for_mask_tracking(model)

        result = model.forward_image(None, need_sam3_out=True, need_interactive_out=True, need_propagation_out=True)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(model.calls[0]["need_sam3_out"], False)
        self.assertEqual(model.calls[0]["need_interactive_out"], True)
        self.assertEqual(model.calls[0]["need_propagation_out"], True)


if __name__ == "__main__":
    unittest.main()

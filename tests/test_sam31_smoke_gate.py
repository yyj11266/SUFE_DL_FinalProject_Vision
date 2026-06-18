from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_sam31_vos import _run_full_quality_gate, _run_smoke_quality_gate


def _write_mask(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)
    return str(path)


class Sam31SmokeGateTest(unittest.TestCase):
    def test_tiny_single_object_empty_later_frames_warns_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks" / "tiny"
            first = np.zeros((100, 100), dtype=np.uint8)
            first[5:7, 5:7] = 1
            empty = np.zeros((100, 100), dtype=np.uint8)
            status = {
                "videos": {
                    "tiny": {
                        "status": "done",
                        "object_ids": [1],
                        "first_frame_exact": True,
                        "mask_paths": [
                            _write_mask(masks / "00000.png", first),
                            _write_mask(masks / "00001.png", first),
                            _write_mask(masks / "00002.png", empty),
                        ],
                        "overlay_paths": [],
                    }
                }
            }

            gate = _run_smoke_quality_gate(root, status)

            self.assertTrue(gate["passed"], gate)
            self.assertEqual(gate["errors"], [])
            self.assertTrue(any("tiny single-object" in warning for warning in gate["warnings"]))

    def test_multi_object_empty_later_frame_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks" / "multi"
            first = np.zeros((20, 20), dtype=np.uint8)
            first[2:8, 2:8] = 1
            first[10:15, 10:16] = 2
            empty = np.zeros((20, 20), dtype=np.uint8)
            status = {
                "videos": {
                    "multi": {
                        "status": "done",
                        "object_ids": [1, 2],
                        "first_frame_exact": True,
                        "mask_paths": [
                            _write_mask(masks / "00000.png", first),
                            _write_mask(masks / "00001.png", empty),
                        ],
                        "overlay_paths": [],
                    }
                }
            }

            gate = _run_smoke_quality_gate(root, status)

            self.assertFalse(gate["passed"], gate)
            self.assertTrue(any("multi-object smoke collapsed" in error for error in gate["errors"]))

    def test_object_diagnostics_early_permanent_zero_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks" / "diagnostic"
            first = np.zeros((20, 20), dtype=np.uint8)
            first[2:8, 2:8] = 1
            later = first.copy()
            status = {
                "videos": {
                    "diagnostic": {
                        "status": "done",
                        "object_ids": [1],
                        "first_frame_exact": True,
                        "mask_paths": [
                            _write_mask(masks / "00000.png", first),
                            _write_mask(masks / "00001.png", later),
                            _write_mask(masks / "00002.png", later),
                        ],
                        "overlay_paths": [],
                        "diagnostics": {
                            "per_object": {
                                "1": {
                                    "total_frames": 20,
                                    "zero_ratio": 0.90,
                                    "first_zero_frame": 3,
                                    "recovers_after_zero": False,
                                    "missing_output_frames": 0,
                                }
                            }
                        },
                    }
                }
            }

            gate = _run_smoke_quality_gate(root, status)

            self.assertFalse(gate["passed"], gate)
            self.assertTrue(any("never recovers" in error for error in gate["errors"]))

    def test_make_submission_requires_sample_submission(self) -> None:
        from scripts.run_sam31_vos import main

        with self.assertRaises(SystemExit) as raised:
            main(["--data-root", "/tmp/missing", "--output-dir", "/tmp/out", "--make-submission"])

        self.assertIn("--sample-submission", str(raised.exception))

    def test_full_quality_gate_fails_extra_empty_masks_vs_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current"
            baseline = root / "baseline"
            first = np.ones((10, 10), dtype=np.uint8)
            empty = np.zeros((10, 10), dtype=np.uint8)
            mask_paths = [
                _write_mask(current / "masks" / "v1" / "00000.png", first),
                _write_mask(current / "masks" / "v1" / "00001.png", empty),
                _write_mask(current / "masks" / "v1" / "00002.png", empty),
            ]
            _write_mask(baseline / "masks" / "v1" / "00000.png", first)
            _write_mask(baseline / "masks" / "v1" / "00001.png", first)
            _write_mask(baseline / "masks" / "v1" / "00002.png", first)
            status = {
                "videos": {
                    "v1": {
                        "status": "done",
                        "object_ids": [1],
                        "mask_paths": mask_paths,
                        "diagnostics": {"per_object": {}},
                    }
                }
            }

            gate = _run_full_quality_gate(
                current,
                status,
                baseline,
                max_extra_empty_frames=1,
                max_extra_empty_ratio=0.0,
                severe_zero_ratio=0.95,
                early_frame_window=20,
            )

            self.assertFalse(gate["passed"], gate)
            self.assertTrue(any("extra non-first empty masks" in error for error in gate["errors"]))


if __name__ == "__main__":
    unittest.main()

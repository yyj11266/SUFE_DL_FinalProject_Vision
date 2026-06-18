from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.mosev2_split import build_mosev2_split


class MoseSplitTest(unittest.TestCase):
    def _dataset(self, root: Path, videos: int = 10) -> None:
        for index in range(videos):
            video_id = f"video_{index:02d}"
            frame_dir = root / "JPEGImages" / video_id
            mask_dir = root / "Annotations" / video_id
            frame_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            object_count = 1 if index % 2 == 0 else 3
            for frame_index in range(3 + index % 3):
                Image.fromarray(np.full((20, 24, 3), index, dtype=np.uint8)).save(frame_dir / f"{frame_index:05d}.jpg")
                mask = np.zeros((20, 24), dtype=np.uint8)
                if not (index % 3 == 0 and frame_index == 2):
                    for object_id in range(1, object_count + 1):
                        offset = object_id * 2
                        mask[offset : offset + 2 + index % 2, offset : offset + 2] = object_id
                Image.fromarray(mask).save(mask_dir / f"{frame_index:05d}.png")

    def test_split_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            first = build_mosev2_split(root, total=8, calibration_size=4, seed=2026)
            second = build_mosev2_split(root, total=8, calibration_size=4, seed=2026)
            self.assertEqual(first["calibration"], second["calibration"])
            self.assertEqual(first["holdout"], second["holdout"])
            self.assertEqual(len(first["calibration"]), 4)
            self.assertEqual(len(first["holdout"]), 4)
            self.assertFalse(set(first["calibration"]) & set(first["holdout"]))


if __name__ == "__main__":
    unittest.main()

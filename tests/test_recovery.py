from __future__ import annotations

import unittest

from src.vos.recovery import (
    ObjectRecoveryController,
    RecoveryBranch,
    RecoveryConfig,
    RecoveryObservation,
    RecoveryState,
)


class RecoveryControllerTest(unittest.TestCase):
    def test_lost_frames_do_not_enter_memory(self) -> None:
        controller = ObjectRecoveryController(3, RecoveryConfig(lost_patience=2))
        weak = RecoveryObservation(4, 0.1, 0.2, 0.4, 5.0, 0.6, 0)
        first = controller.step(weak)
        second = controller.step(RecoveryObservation(5, 0.1, 0.2, 0.4, 5.0, 0.6, 0))
        self.assertEqual(first.state, RecoveryState.AMBIGUOUS)
        self.assertEqual(second.state, RecoveryState.LOST)
        self.assertFalse(first.allow_memory_write)
        self.assertFalse(second.allow_memory_write)
        self.assertEqual(controller.confirmed_memory_frames, [0])

    def test_branch_count_is_bounded_and_confirmation_recovers(self) -> None:
        controller = ObjectRecoveryController(1, RecoveryConfig(max_branches=3, recovery_confirm_frames=2))
        for index in range(5):
            controller.add_branch(RecoveryBranch(str(index), "motion_box", index / 10, 7, 0.8, 0.8))
        self.assertEqual(len(controller.branches), 3)
        branch_id = next(iter(controller.branches))
        self.assertFalse(
            controller.update_branch(
                branch_id,
                frame_index=8,
                score=0.9,
                identity_similarity=0.9,
                position_consistency=0.8,
            )
        )
        self.assertTrue(
            controller.update_branch(
                branch_id,
                frame_index=9,
                score=0.9,
                identity_similarity=0.9,
                position_consistency=0.8,
            )
        )
        decision = controller.step(
            RecoveryObservation(9, 0.8, 0.8, 0.9, 1.1, 0.0, 20),
            confirmed_branch_id=branch_id,
        )
        self.assertEqual(decision.state, RecoveryState.RECOVERED)
        self.assertTrue(decision.allow_memory_write)


if __name__ == "__main__":
    unittest.main()

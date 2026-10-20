import unittest
from drivable_area.tracking.direction_votes import DirectionVoteTracker


def ev(label):
    return {"direction": label}


class DirectionVoteTests(unittest.TestCase):
    def test_predictions_do_not_inflate_observation_count_or_run(self):
        t = DirectionVoteTracker(60)
        t.update(ev("left"), True)
        for _ in range(4):
            r = t.update(ev("left"), False)
        self.assertEqual(r["total_observations"], 1)
        self.assertEqual(r["consecutive_observations"], 1)
        self.assertEqual(r["direction"], "unknown")

    def test_consistent_observations_confirm_after_minimum_run(self):
        t = DirectionVoteTracker(15)
        for _ in range(3):
            r = t.update(ev("straight"), True)
        self.assertEqual(r["direction"], "straight")
        self.assertAlmostEqual(sum(r["weights"].values()), 1)
        self.assertEqual(r["agreement_count"], 3)

    def test_a_single_outlier_never_changes_the_stable_direction(self):
        t = DirectionVoteTracker(15)
        for _ in range(8):
            t.update(ev("straight"), True)
        r = t.update(ev("right"), True)
        self.assertEqual(r["direction"], "straight")
        self.assertEqual(r["pending_direction"], "right")
        self.assertLess(r["weights"]["right"], r["weights"]["straight"])

    def test_a_real_turn_eventually_switches_with_fresh_support(self):
        t = DirectionVoteTracker(60)
        for i in range(60):
            t.update(ev("straight"), i % 5 == 0)
        switched = False
        for i in range(60):
            r = t.update(ev("right"), i % 5 == 0)
            if r["direction"] == "right":
                switched = True
                break
        self.assertTrue(switched)
        self.assertGreaterEqual(r["consecutive_observations"], 3)

    def test_unknown_observations_reduce_agreement_and_expire_old_label(self):
        t = DirectionVoteTracker(10)
        for _ in range(3):
            t.update(ev("left"), True)
        r = t.update(ev("unknown"), True)
        self.assertEqual(r["state"], "held")
        self.assertLess(r["agreement_fraction"], 1)
        for _ in range(7):
            r = t.update(ev("unknown"), True)
        self.assertEqual(r["direction"], "unknown")
        self.assertTrue(r["stale"])

    def test_alternating_observations_cannot_confirm_a_turn(self):
        t = DirectionVoteTracker(15)
        for label in ["left", "right"] * 10:
            r = t.update(ev(label), True)
        self.assertEqual(r["direction"], "unknown")

    def test_window_bounds_history_and_arrow_weights_remain_bounded(self):
        t = DirectionVoteTracker(30)
        for i in range(150):
            r = t.update(ev("right"), i % 3 == 0)
        self.assertLessEqual(r["recent_observations"], 11)
        self.assertTrue(all(0 <= p <= 1 for p in r["arrow_weights"].values()))

import unittest

from log_barrier.lunar_barrier.statistics import (
    paired_reliability_summary,
    rank_configurations,
    select_barrier_configuration,
)


def _result(seed, method, value, learning_rate=0.001, beta=0.0):
    return {
        "seed": seed,
        "method": method,
        "stochastic_evaluation_mean": value,
        "stochastic_evaluation_returns": [value] * 32,
        "config": {"learning_rate": learning_rate, "beta": beta},
    }


class LunarBarrierStatisticsTests(unittest.TestCase):
    def test_smoke_selection_uses_lower_tail(self):
        results = [
            _result(seed, "reward_only", value, beta=beta)
            for beta, values in ((1.0, (-100.0, 100.0, 100.0)), (10.0, (20.0, 20.0, 20.0)))
            for seed, value in enumerate(values, 1)
        ]
        ranking = rank_configurations(results, ("beta",), lower_quantile=0.25)
        self.assertEqual(ranking[0]["beta"], 10.0)

    def test_barrier_selection_is_paired_to_baseline(self):
        baselines = []
        for seed, baseline in ((1, 0.0), (2, 0.0), (3, 0.0)):
            baselines.append(_result(seed, "reward_only", baseline))
        results = []
        for seed, value in enumerate((-100.0, 100.0, 100.0), 1):
            row = _result(seed, "log_barrier", value, beta=1.0)
            row["config"]["handoff_fraction"] = 0.25
            results.append(row)
        for seed, value in enumerate((20.0, 20.0, 20.0), 1):
            row = _result(seed, "log_barrier", value, beta=10.0)
            row["config"]["handoff_fraction"] = 0.25
            results.append(row)
        selection = select_barrier_configuration(baselines, results, 0.25)
        self.assertEqual(selection["selected"]["beta"], 10.0)

    def test_paired_summary_uses_within_seed_differences(self):
        results = [
            _result(1, "reward_only", -100.0),
            _result(1, "log_barrier", -90.0, beta=10.0),
            _result(2, "reward_only", 200.0),
            _result(2, "log_barrier", 220.0, beta=10.0),
        ]
        summary = paired_reliability_summary(results, bootstrap_samples=100, bootstrap_seed=1)
        paired = summary["paired_difference_log_barrier_minus_reward_only"]
        self.assertEqual(paired["mean"], 15.0)
        self.assertEqual(paired["win_rate"], 1.0)
        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual(summary["evaluation_episodes_per_policy"], 32)

    def test_incomplete_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_reliability_summary(
                [_result(1, "reward_only", 0.0), _result(2, "log_barrier", 0.0)]
            )


if __name__ == "__main__":
    unittest.main()

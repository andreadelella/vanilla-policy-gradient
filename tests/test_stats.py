import unittest

import numpy as np

from vpg.stats import mean_confidence_interval, student_t_critical_95


class ConfidenceIntervalTests(unittest.TestCase):
    def test_five_seed_interval_uses_four_degrees_of_freedom(self):
        values = np.arange(1.0, 6.0)[:, None]
        mean, lower, upper = mean_confidence_interval(values)
        margin = 2.7764451052 * values[:, 0].std(ddof=1) / np.sqrt(5)

        np.testing.assert_allclose(mean, [3.0])
        np.testing.assert_allclose(lower, [3.0 - margin])
        np.testing.assert_allclose(upper, [3.0 + margin])

    def test_interval_requires_independent_repeats(self):
        with self.assertRaises(ValueError):
            mean_confidence_interval([[1.0]])

    def test_critical_value_requires_two_samples(self):
        with self.assertRaises(ValueError):
            student_t_critical_95(1)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from scipy.special import expit

from dlshogi.evalfix import EVAL_COEFFICIENT as K, fit_evalfix


class AffineEvalfixTest(unittest.TestCase):
    def test_recovers_scale_and_offset(self):
        scores = np.linspace(-1800, 1800, 200)
        for scale, offset in ((1.8, -150.0), (0.6, 200.0), (1.0, 0.0)):
            with self.subTest(scale=scale, offset=offset):
                a, fitted_offset = fit_evalfix(scores, expit((scores * scale + offset) / K))
                self.assertAlmostEqual(K / a, scale, places=5)
                self.assertAlmostEqual(fitted_offset, offset, places=3)

    def test_game_results_with_draws(self):
        scores = np.repeat([-600, 0, 600], 100)
        results = np.concatenate([
            np.array([1] * wins + [0.5] * 20 + [0] * (80 - wins))
            for wins in (10, 30, 60)
        ])
        a, offset = fit_evalfix(scores, results)
        self.assertGreater(a, 0)
        self.assertLess(offset, 0)
        self.assertLess(abs(expit(offset / K) - 0.4), 0.04)

    def test_degenerate_files_are_unchanged(self):
        for scores in ([], [100], [0, 0], [100, 100]):
            self.assertEqual(fit_evalfix(scores, np.zeros(len(scores))), (0, 0))

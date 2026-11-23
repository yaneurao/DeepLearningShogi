"""Affine score calibration against side-to-move game results."""

import numpy as np

EVAL_COEFFICIENT = 756.0864962951762


def fit_evalfix(scores, results):
    """Return (a, offset) for corrected_score = score * K / a + offset.

    Keep the existing least-squares probability objective. Constant scores
    cannot identify both parameters, so leave those files unmodified.
    """
    from scipy.optimize import curve_fit
    from scipy.special import expit

    scores = np.asarray(scores, dtype=np.float64)
    results = np.asarray(results, dtype=np.float64)
    if scores.size < 2 or np.ptp(scores) == 0:
        return 0.0, 0.0

    def probability(x, scale, bias):
        return expit(x * scale + bias)

    (scale, bias), _ = curve_fit(
        probability, scores / EVAL_COEFFICIENT, results,
        p0=[EVAL_COEFFICIENT / 300.0, 0.0],
        bounds=([1e-8, -np.inf], [np.inf, np.inf]),
    )
    return EVAL_COEFFICIENT / scale, EVAL_COEFFICIENT * bias

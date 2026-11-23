"""Integration tests: build cppshogi first; no Torch/GPU required."""
import struct
import tempfile
import unittest
from pathlib import Path

import cshogi
import numpy as np

from dlshogi import cppshogi
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.evalfix import EVAL_COEFFICIENT as K


class NativeEvalfixTest(unittest.TestCase):
    def test_affine_values_in_both_teacher_formats(self):
        board = cshogi.Board()
        positions = []
        for _ in range(2):
            hcp = np.empty(32, dtype=np.uint8)
            board.to_hcp(hcp)
            move = next(iter(board.legal_moves))
            positions.append((hcp.tobytes(), cshogi.move16(move)))
            board.push(move)

        for suffix in ('hcpe', 'hcpe3'):
            for scale, offset in ((1.5, -125.0), (0.75, 220.0), (1.0, 0.0)):
                with self.subTest(suffix=suffix, scale=scale, offset=offset), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / ('input.' + suffix)
                    scores = [-300, 500]
                    with path.open('wb') as out:
                        for (hcp, move), score in zip(positions, scores):
                            if suffix == 'hcpe':
                                out.write(hcp + struct.pack('<hHBB', score, move, 1, 0))
                            else:
                                out.write(hcp + struct.pack('<HBB', 1, 1, 0))
                                out.write(struct.pack('<HhHHH', move, score, 1, move, 10))
                    size, count = cppshogi.load_hcpe3(str(path), False, K / scale, 1.0, 1.0, offset)
                    self.assertEqual(count, 2)
                    features1 = np.empty((2, FEATURES1_NUM, 9, 9), np.float32)
                    features2 = np.empty((2, FEATURES2_NUM, 9, 9), np.float32)
                    policy = np.empty((2, MAX_MOVE_LABEL_NUM * 81), np.float32)
                    result = np.empty((2, 1), np.float32)
                    value = np.empty((2, 1), np.float32)
                    cppshogi.hcpe3_decode_with_value(
                        np.arange(size - count, size, dtype=np.uint64),
                        features1, features2, policy, result, value)
                    expected = 1 / (1 + np.exp(-(np.array(scores) * scale + offset) / K))
                    np.testing.assert_allclose(value[:, 0], expected, atol=1e-6)
                    np.testing.assert_array_equal(result[:, 0], [1, 0])
                    np.testing.assert_allclose(policy.sum(axis=1), [1, 1])

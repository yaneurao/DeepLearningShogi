import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_case(temperature, mix, average, hcpe=False):
    import cshogi
    from cshogi.dlshogi import make_move_label
    import numpy as np
    from dlshogi import cppshogi
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

    board = cshogi.Board()
    moves = [board.move_from_usi(s) for s in ['7g7f', '2g2f']]
    move16 = [m & 65535 for m in moves]
    hcp = np.zeros(1, dtype=cshogi.HuffmanCodedPos)
    board.to_hcp(hcp)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ('data.hcpe' if hcpe else 'data.hcpe3')
        if hcpe:
            records = np.zeros(2, dtype=cshogi.HuffmanCodedPosAndEval)
            for i in range(2):
                records['hcp'][i] = hcp['hcp'][0]
                records['eval'][i] = 100
                records['bestMove16'][i] = move16[i]
                records['gameResult'][i] = 1
            path.write_bytes(records.tobytes())
        else:
            raw = bytearray()
            for selected in move16:
                raw += hcp.tobytes() + struct.pack('<HBB', 1, 1, 0)
                raw += struct.pack('<HhH', selected, 100, 2)
                raw += struct.pack('<HHHH', move16[0], 100, move16[1], 300)
            path.write_bytes(raw)
        args = (str(path), average, 0, temperature)
        n, actual = cppshogi.load_hcpe3(*args) if mix is None else cppshogi.load_hcpe3(*args, mix)
    assert actual == 2 and n == (1 if average else 2), (n, actual)
    f1 = np.zeros((n, FEATURES1_NUM, 9, 9), np.float32)
    f2 = np.zeros((n, FEATURES2_NUM, 9, 9), np.float32)
    p = np.zeros((n, 2187), np.float32)
    result = np.zeros(n, np.float32)
    value = np.zeros(n, np.float32)
    cppshogi.hcpe3_decode_with_value(np.arange(n, dtype=np.uint64), f1, f2, p, result, value)
    visits = np.array([100, 300], dtype=float)
    if temperature == 0:
        q = np.array([0., 1.])
    else:
        q = (visits / visits.max()) ** (1 / temperature)
        q /= q.sum()
    alpha = 1 if mix is None else mix
    expected = np.eye(2) if hcpe else alpha*q + (1-alpha)*np.eye(2)
    if average:
        expected = expected.mean(axis=0, keepdims=True)
    labels = [make_move_label(m, board.turn) for m in moves]
    np.testing.assert_allclose(p[:, labels], expected, atol=1e-7)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(result, 1)
    np.testing.assert_allclose(value, 1/(1+np.exp(-100*.0013226)), atol=1e-7)


class PolicyMixTest(unittest.TestCase):
    def test_targets_and_duplicate_averaging(self):
        for temperature in (0, .1, 1):
            for mix in (None, 0, .25, 1):
                for average in (False, True):
                    with self.subTest(temperature=temperature, mix=mix, average=average):
                        subprocess.run([sys.executable, __file__, '--case',
                                        json.dumps([temperature, mix, average])], check=True)

    def test_hcpe_targets_are_unchanged(self):
        for mix in (0, .25, 1):
            subprocess.run([sys.executable, __file__, '--case', json.dumps([1, mix, True, True])], check=True)

    def test_invalid_mix(self):
        from dlshogi import cppshogi
        for mix in (-1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                cppshogi.load_hcpe3('unused', False, 0, 1, mix)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--case':
        check_case(*json.loads(sys.argv[2]))
    else:
        unittest.main()

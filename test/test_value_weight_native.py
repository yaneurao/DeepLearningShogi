import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_case(average, cached, hcpe):
    import cshogi
    import numpy as np
    from dlshogi import cppshogi
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
    from dlshogi.data_loader import Hcpe3DataLoader

    board = cshogi.Board()
    hcp = np.zeros(1, dtype=cshogi.HuffmanCodedPos)
    board.to_hcp(hcp)
    move = board.move_from_usi('7g7f') & 65535
    scores = [-30000, 30000, 0, -400, 900]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ('teacher.hcpe' if hcpe else 'teacher.hcpe3')
        if hcpe:
            records = np.zeros(len(scores), dtype=cshogi.HuffmanCodedPosAndEval)
            records['hcp'] = hcp['hcp'][0]
            records['eval'] = scores
            records['bestMove16'] = move
            records['gameResult'] = 1
            path.write_bytes(records.tobytes())
        else:
            path.write_bytes(b''.join(hcp.tobytes() + struct.pack('<HBBHhHHH', 1, 1, 0, move, score, 1, move, 100)
                                      for score in scores))
        n, _ = cppshogi.load_hcpe3(str(path), average, 600., 1.)
        if cached:
            cache = str(Path(tmp) / 'cache')
            cppshogi.hcpe3_create_cache(cache)
            cppshogi.hcpe3_load_cache(cache)
        indices = np.arange(n, dtype=np.uint64)[::-1].copy()
        f1 = np.zeros((n, FEATURES1_NUM, 9, 9), np.float32)
        f2 = np.zeros((n, FEATURES2_NUM, 9, 9), np.float32)
        p = np.zeros((n, 81 * MAX_MOVE_LABEL_NUM), np.float32)
        result = np.zeros(n, np.float32)
        values = np.zeros(n, np.float32)
        cppshogi.hcpe3_decode_with_value(indices, f1, f2, p, result, values)
        loader = Hcpe3DataLoader.__new__(Hcpe3DataLoader)
        loader.data = indices
        for minimum in (0., .25, .5, 1.):
            expected = (minimum + (1-minimum)*4*values*(1-values)).mean()
            np.testing.assert_allclose(cppshogi.hcpe3_value_weight_mean(indices, minimum), expected, atol=1e-7)
            subset = values[-1:]
            expected = (minimum + (1-minimum)*4*subset*(1-subset)).mean()
            np.testing.assert_allclose(loader.value_weight_mean(n-1, 1, minimum), expected, atol=1e-7)
        for invalid in (-.1, 1.1, float('nan'), float('inf')):
            try:
                cppshogi.hcpe3_value_weight_mean(indices, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid minimum accepted')
        try:
            cppshogi.hcpe3_value_weight_mean(np.array([n], np.uint64), .5)
        except IndexError:
            pass
        else:
            raise AssertionError('invalid index accepted')
        try:
            cppshogi.hcpe3_value_weight_mean(np.array([], np.uint64), .5)
        except ValueError:
            pass
        else:
            raise AssertionError('empty indices accepted')


class NativeWeightTest(unittest.TestCase):
    def test_same_values_as_decoder(self):
        for average in (False, True):
            for cached in (False, True):
                for hcpe in (False, True):
                    with self.subTest(average=average, cached=cached, hcpe=hcpe):
                        subprocess.run([sys.executable, __file__, '--case', json.dumps([average, cached, hcpe])], check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--case':
        check_case(*json.loads(sys.argv[2]))
    else:
        unittest.main()

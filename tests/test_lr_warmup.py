import ast
from pathlib import Path
import unittest


class WarmupTest(unittest.TestCase):
    def test_endpoints_and_monotonicity(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'dlshogi/train.py').read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'warmup_lr')
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), '<warmup>', 'exec'), namespace)
        lr = namespace['warmup_lr']
        for updates in (1, 2, 3, 100):
            values = [lr(.00001, .001, i, updates) for i in range(updates)]
            self.assertEqual(values, sorted(values))
            self.assertAlmostEqual(values[-1], .001)
            self.assertAlmostEqual(values[0], .001 if updates == 1 else .00001)

    def test_both_update_paths_and_scheduler_guards(self):
        source = (Path(__file__).resolve().parents[1] / 'dlshogi/train.py').read_text()
        tree = ast.parse(source)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertEqual(sum(isinstance(n.func, ast.Name) and n.func.id == 'apply_warmup_lr' for n in calls), 2)
        self.assertEqual(source.count("and not warming_up:"), 3)

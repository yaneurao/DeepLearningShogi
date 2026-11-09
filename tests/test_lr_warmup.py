import ast
import argparse
from pathlib import Path
import unittest


class WarmupTest(unittest.TestCase):
    def test_optional_argument(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'dlshogi/train.py').read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == '--lr-warmup')
        parser = argparse.ArgumentParser()
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)],
                                                         type_ignores=[])), '<parser>', 'exec'),
             {'parser': parser})
        self.assertIsNone(parser.parse_args([]).lr_warmup)
        self.assertEqual(parser.parse_args(['--lr-warmup']).lr_warmup, 1e-7)
        self.assertEqual(parser.parse_args(['--lr-warmup', '0']).lr_warmup, 0)

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

"""Exercise the production loader method without requiring Torch or a GPU."""
import ast
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


class EvalfixATest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'dlshogi/data_loader.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Hcpe3DataLoader')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_files')
        method.decorator_list = []
        self.cpp = Mock()
        self.cpp.load_hcpe3.return_value = (10, 10)
        namespace = dict(np=np, os=os, logging=logging, cppshogi=self.cpp,
                         score_to_value=lambda score, a: 1 / (1 + np.exp(-score / a)))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        self.load = namespace['load_files']

    def test_fixed_skips_fitting_and_passes_coefficient_for_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extension in ('hcpe', 'hcpe3'):
                path = Path(tmp) / ('input.' + extension)
                path.touch()
                for enabled in (False, True):
                    self.assertEqual(self.load([str(path)], use_evalfix=enabled, evalfix_a=600), (10, 10))
                    self.cpp.load_hcpe3.assert_called_with(str(path), False, 600, 1.0)
                    self.cpp.hcpe3_prepare_evalfix.assert_not_called()

    def test_unspecified_off_unchanged(self):
        with tempfile.NamedTemporaryFile() as f:
            self.load([f.name])
            self.cpp.load_hcpe3.assert_called_with(f.name, False, 0, 1.0)

    def test_unspecified_on_still_fits(self):
        import types
        import sys
        curve = Mock(return_value=([150.0], None))
        optimize = types.ModuleType('scipy.optimize')
        optimize.curve_fit = curve
        self.cpp.hcpe3_prepare_evalfix.return_value = (np.array([1, 2]), np.array([0, 1]))
        with patch.dict(sys.modules, {'scipy.optimize': optimize}), tempfile.NamedTemporaryFile() as f:
            self.load([f.name], use_evalfix=True)
            curve.assert_called_once()
            self.cpp.load_hcpe3.assert_called_with(f.name, False, 150.0, 1.0)

    def test_validation(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load([], evalfix_a=value)
        for option in ('cache', 'patch'):
            with self.assertRaises(ValueError):
                self.load([], evalfix_a=600, **{option: 'file'})
        self.cpp.load_hcpe3.assert_not_called()


if __name__ == '__main__':
    unittest.main()

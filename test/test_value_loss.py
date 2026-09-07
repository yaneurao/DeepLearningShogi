import copy
import importlib
import io
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dlshogi.value_loss import weighted_value_losses, positive_denominator, ValueGradientAccumulator


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(3, 4)
        self.policy = torch.nn.Linear(4, 2)
        self.value = torch.nn.Linear(4, 1)

    def forward(self, x, unused=None):
        hidden = self.shared(x).tanh()
        return self.policy(hidden), self.value(hidden)


class ValueLossTest(unittest.TestCase):
    def test_train_main_updates_match_reference_and_test_loss_is_unweighted(self):
        # Native shogi decoding is replaced with tiny deterministic batches only.
        native = ModuleType('dlshogi.cppshogi')
        native.get_max_features2_nyugyoku_num = lambda: 0
        with patch.dict(sys.modules, {'dlshogi.cppshogi': native}):
            train = importlib.import_module('dlshogi.train')
        torch.manual_seed(77)
        x = torch.randn(8, 3)
        q = torch.tensor([[0.], [1.], [.4], [.5], [.6], [.8], [.2], [.9]])
        r = torch.tensor([[0.], [1.], [0.], [1.], [1.], [0.], [1.], [0.]])
        labels = torch.tensor([0, 1, 0, 1, 1, 1, 0, 0])

        class TrainLoader:
            @staticmethod
            def load_files(*args, **kwargs):
                return 8, 8

            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                for i in range(0, 8, 2):
                    yield x[i:i+2], x[i:i+2], F.one_hot(labels[i:i+2], 2).float(), r[i:i+2], q[i:i+2]

        class TestLoader(TrainLoader):
            def __iter__(self):
                yield x, x, labels, r, q

        for batches in (1, 3):
            for minimum in (0., .5, 1.):
                with self.subTest(batches=batches, minimum=minimum), tempfile.TemporaryDirectory() as tmp:
                    model = TinyNetwork()
                    reference = copy.deepcopy(model)
                    optimizer = torch.optim.SGD(reference.parameters(), lr=.02)
                    for start in range(0, 8, batches * 2):
                        end = min(8, start + batches * 2)
                        py, vy = reference(x[start:end])
                        l2, l3, w = weighted_value_losses(vy, r[start:end], q[start:end], minimum)
                        loss = F.cross_entropy(py, labels[start:end]) + (.6*l2 + .4*l3) / positive_denominator(w)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    with patch.object(train, 'policy_value_network', return_value=model), \
                            patch.object(train, 'Hcpe3DataLoader', TrainLoader), \
                            patch.object(train, 'DataLoader', TestLoader), \
                            patch.object(train.np, 'fromfile', return_value=train.np.zeros(8)), \
                            patch.object(train.logging, 'info') as log:
                        train.main('train.hcpe3', 'test.hcpe', '--gpu', '-1', '--batchsize', '2',
                                   '--batches-per-update', str(batches), '--value-loss-min-weight', str(minimum),
                                   '--optimizer', 'SGD()', '--weight_decay', '0', '--lr', '.02',
                                   '--val_lambda', '.4', '--clip_grad_max_norm', '0', '--checkpoint', '')
                    for actual, expected in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
                    with torch.no_grad():
                        py, vy = model(x)
                        total = F.cross_entropy(py, labels) + .6*F.binary_cross_entropy_with_logits(vy, r) + .4*F.binary_cross_entropy_with_logits(vy, q)
                    final_log = [call.args[0] for call in log.call_args_list if 'train loss avr' in str(call.args[0])][-1]
                    reported = float(final_log.split('test loss = ')[1].split(', ')[3])
                    self.assertAlmostEqual(reported, total.item(), places=6)
        for invalid in ('-1', '1.01', 'nan', 'inf'):
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                train.main('train.hcpe3', 'test.hcpe', '--value-loss-min-weight', invalid)

    def test_formula_and_detached_teacher(self):
        q = torch.tensor([[0.], [.25], [.5], [.75], [1.]], requires_grad=True)
        logits = torch.zeros_like(q, requires_grad=True)
        loss2, loss3, weight = weighted_value_losses(logits, torch.zeros_like(q), q, .5)
        expected_weights = torch.tensor([.5, .875, 1., .875, .5])
        torch.testing.assert_close(weight, expected_weights.mean())
        torch.testing.assert_close(loss2, expected_weights.mean() * torch.log(torch.tensor(2.)))
        loss3.backward()
        self.assertIsNone(q.grad)

    def test_accumulated_gradients_match_concatenated_batch(self):
        torch.manual_seed(31)
        x = torch.randn(6, 3)
        q = torch.tensor([[0.], [1.], [.49], [.51], [.15], [.9]])
        result = torch.tensor([[0.], [1.], [1.], [0.], [1.], [0.]])
        labels = torch.tensor([0, 1, 1, 0, 0, 1])
        for minimum in (0., .25, .5, 1.):
            for count in (1, 2, 3):
                for scaled in (False, True):
                    with self.subTest(minimum=minimum, batches=count, scaled=scaled):
                        model = TinyNetwork()
                        reference = copy.deepcopy(model)
                        n = count * 2
                        py, vy = reference(x[:n])
                        l2, l3, w = weighted_value_losses(vy, result[:n], q[:n], minimum)
                        expected_loss = F.cross_entropy(py, labels[:n]) + (.6*l2 + .4*l3) / positive_denominator(w)
                        expected_loss.backward()
                        scaler = torch.amp.GradScaler('cpu', enabled=scaled)
                        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
                        accumulator = ValueGradientAccumulator(model.parameters())
                        for start in range(0, n, 2):
                            end = start + 2
                            py, vy = model(x[start:end])
                            l2, l3, w = weighted_value_losses(vy, result[start:end], q[start:end], minimum)
                            accumulator.backward(F.cross_entropy(py, labels[start:end]),
                                                 .6*l2 + .4*l3, w, count, scaler)
                        accumulator.finish(count)
                        scaler.unscale_(optimizer)
                        for actual, expected in zip(model.parameters(), reference.parameters()):
                            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-7, rtol=2e-5)

    def test_one_weight_is_original_bce(self):
        logits = torch.tensor([[.5], [-2.], [3.]])
        q = torch.tensor([[.2], [.5], [.8]])
        results = torch.tensor([[0.], [1.], [0.]])
        l2, l3, weight = weighted_value_losses(logits, results, q, 1.)
        self.assertEqual(weight.item(), 1.)
        torch.testing.assert_close(l2, F.binary_cross_entropy_with_logits(logits, results))
        torch.testing.assert_close(l3, F.binary_cross_entropy_with_logits(logits, q))


if __name__ == '__main__':
    unittest.main()

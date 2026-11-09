import copy
import unittest

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.policy_value_network_exp__by import PolicyValueNetwork


class ActivationCheckpointTest(unittest.TestCase):
    def test_gradients_and_batchnorm_running_statistics(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        ordinary = PolicyValueNetwork(2, 8, nhead=2, fcl=8).train()
        checkpointed = copy.deepcopy(ordinary)
        checkpointed.activation_checkpoint = True
        inputs = (torch.randn(2, FEATURES1_NUM, 9, 9),
                  torch.randn(2, FEATURES2_NUM, 9, 9))
        for model in (ordinary, checkpointed):
            sum(output.square().mean() for output in model(*inputs)).backward()
        for (name, first), (other_name, second) in zip(ordinary.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(first.grad, second.grad)
        for (name, first), (other_name, second) in zip(ordinary.named_buffers(), checkpointed.named_buffers()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(first, second)


if __name__ == '__main__':
    unittest.main()

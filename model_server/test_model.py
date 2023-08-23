import unittest
import torch
from model import ConnNet
class TestModel(unittest.TestCase):
    def test_run(self):
        tops = 3
        els = 5
        net = ConnNet(tops,4)
        inputs = torch.rand((els,2,tops,4))
        policy, value = net(inputs)
        self.assertEqual(tuple(policy.shape), (els, tops))
        self.assertEqual(tuple(value.shape), (els, 1))


if __name__ == '__main__':
    unittest.main()

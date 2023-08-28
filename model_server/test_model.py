import unittest
import torch
from unittest import skip
from model import ConnNet
import time
class TestModel(unittest.TestCase):

    @skip
    def test_run(self):
        tops = 3
        els = 5
        net = ConnNet(tops,4)
        inputs = torch.rand((els,2,tops,4))
        policy, value = net(inputs)
        self.assertEqual(tuple(policy.shape), (els, tops))
        self.assertEqual(tuple(value.shape), (els, 1))
    
    @skip
    def test_inference_speed(self):
        tops = 4
        rows = 3
        els = 9
        device = torch.device('cuda')
        net = ConnNet(tops,rows)
        net.to(device)
        times = list()
        for i in range(4):
            start = time.time()
            inputs = torch.rand((els,2,tops,rows))
            with torch.no_grad():
                policy, value = net(inputs.to(device))
            dur = time.time()-start
            times.append(dur)
        times = times[1:]
        print(f'Inference in {sum(times)/len(times):.5f} seconds on {device}')
    
    def test_saves(self):
        tops = 4
        rows = 3
        els = 9
        device = torch.device('cuda')
        net = ConnNet(tops,rows)
        net.to(device)
        torch.save(net.state_dict(), '/tmp/net.pth')
        net1 = ConnNet(tops,rows)
        net1.to(device)
        net1.load_state_dict(torch.load('/tmp/net.pth'))
        print(next(net1.parameters()).device)


if __name__ == '__main__':
    unittest.main()

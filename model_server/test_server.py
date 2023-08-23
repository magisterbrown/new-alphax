import unittest
import requests
import numpy as np
import json
import scipy
from unittest import skip
import pretty_errors
import torch

class TestServer(unittest.TestCase):
    url = 'http://localhost:9000'
    fields_size = (3*3)
    def setUp(self):
        
        self.field =  np.random.randint(0,3, self.fields_size).tolist()
    
    @skip
    def test_get(self):
        resp = requests.get(self.url,params={"field": json.dumps(self.field), "my_figure": 1, "enemy_figure": 2})
        print(resp.content)

    def test_post(self):
        print('AAAAAAAAA')
        flds = [
                {'field':self.field, 'probs':{1:3,2:25}, 'player_fig':1, 'enemy_fig': 2},
                {'field':self.field, 'probs':dict(), 'player_fig':2, 'enemy_fig': 1},
                {'field':self.field, 'probs':{0:4,1:4,2:4}, 'player_fig':1, 'enemy_fig': 2},
                ]
        data = {'steps':flds, 'winner':1}
        temp=1e-3
        resp = requests.post(self.url, json = data)
        print(resp.content)


if __name__ == '__main__':
    unittest.main()


import unittest
import requests
import numpy as np
import json
import scipy
from unittest import skip
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
        data = {'steps':[{'field':self.field, 'probs':[{1:3,2:25}], 'player_fig':1}], 'winner':1}
        temp=1e-3
        #resp = requests.post(self.url, json = data)
        probs = data['fields']
        probs = {1:3,2:3}
        COLS = 3
        serialize_in = torch.float32
        mcts_probs = torch.zeros((COLS),dtype=serialize_in)
        mcts_probs[list(probs.keys())] = torch.tensor(scipy.special.softmax(1.0/temp * np.log(np.array(list(probs.values())) + 1e-10)),dtype=serialize_in)
        import ipdb; ipdb.set_trace()

        print(act_probs)

        print(resp.content)


if __name__ == '__main__':
    unittest.main()


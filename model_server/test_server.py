import unittest
import redis
import requests
import numpy as np
import json
import scipy
from unittest import skip
import torch
from server import model_trainer
import random
import time
import os

class TestServer(unittest.TestCase):
    url = 'http://localhost:9000'
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    cols = 4
    rows = 3
    def setUp(self):
        self.fields_size = self.rows*self.cols
        self.field = np.random.randint(0,3, self.fields_size).tolist()
        self.figs = [1,2]

    def gen_random_game(self, steps: int):
        res = list()
        for i in range(steps):
            field = np.random.randint(0,3, self.fields_size).tolist()
            steps = random.sample(range(0, self.cols), random.randint(0, self.cols))
            steps = {step: random.randint(1, 255) for step in steps}
            random.shuffle(self.figs)
            res.append({'field': field, 'probs': steps, 'player_fig':self.figs[0], 'enemy_fig': self.figs[1]})
        return {'steps': res, 'winner': random.choice(self.figs)}
    
    @skip
    def test_get(self):
        resp = requests.get(self.url,params={"field": json.dumps(self.field), "my_figure": 1, "enemy_figure": 2})
        print(resp.content)
    @skip
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
    
    @skip
    def test_train(self):
        self.redis_client.ltrim("to_learn", 1, 0)
        data=self.gen_random_game(8)
        resp = requests.post(self.url, json = data)

        #model_trainer()

    def test_speed_to_save(self):
        datafile = 'checkpoints/latest.pth'
        self.redis_client.ltrim("to_analyze", 1, 0)
        self.redis_client.ltrim("to_learn", 1, 0)
        try: 
            os.remove(datafile) 
        except:
            pass
        start = time.time()
        data=self.gen_random_game(512)
        resp = requests.post(self.url, json = data)
        print('Posted games')
        while not os.path.exists('checkpoints/latest.pth'):
            time.sleep(1)

        print(f'Trained in {time.time()-start:.2f} seconds')

if __name__ == '__main__':
    unittest.main()


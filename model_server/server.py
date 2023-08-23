import os
import json
import torch
import numpy as np
import time
from torch.multiprocessing import Process, Queue, Pipe
from urllib.parse import parse_qs
from dataclasses import dataclass
from multiprocessing.connection import Connection
from model import ConnNet
from typing import List
import redis
import pickle
import base64
import io
import uuid
import scipy
import random

class DataPosition:
    def __init__(self, field: torch.Tensor):
        self.field = field
        self.uuid = uuid.uuid4().hex

@dataclass
class PlayedPosition:
    field: torch.Tensor
    probs: torch.Tensor
    value: torch.Tensor

redis_enc = lambda x: base64.b64encode(pickle.dumps(x))
redis_dec = lambda x: pickle.loads(base64.b64decode(x))
processing_line = 'to_analyze'
training_line = 'to_learn'
learn_batch_size = 2 
batch_size = 16
ROWS = os.environ.get("ROWS", 3)
COLS = os.environ.get("COLS", 3)
DTYPE = torch.float32
serialize_in = DTYPE
redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)
temp = 1e-3

def field_to_tenor(field: List[int], my_fig: int, enemy_fig: int) -> torch.Tensor:
    positions = np.resize(np.array(field), (ROWS, COLS))
    return torch.stack([
            torch.tensor(positions==my_fig, dtype=serialize_in),
            torch.tensor(positions==enemy_fig, dtype=serialize_in)
        ])

def application(env, start_response):
    if(env['REQUEST_METHOD'] == 'GET'):
        game = parse_qs(env['QUERY_STRING'])
        posed = DataPosition(field_to_tenor(json.loads(game['field'][0]), int(game['my_figure']), int(game['enemy_figure'])))
        redisClient.lpush(processing_line, redis_enc(posed))
        _, res = redisClient.blpop(posed.uuid)

    elif(env['REQUEST_METHOD'] == 'POST'):
        data = json.loads(env['wsgi.input'].read().decode('utf-8'))
        results = list()
        for step in data['steps']:
            probs=step['probs']
            mcts_probs = torch.zeros((COLS),dtype=serialize_in)
            if len(probs.keys()):
                mcts_probs[np.array(list(probs.keys()), dtype=np.int64)] = torch.tensor(scipy.special.softmax(1.0/temp * np.log(np.array(list(probs.values())) + 1e-10)),dtype=serialize_in)
            field = field_to_tenor(step['field'], step['player_fig'], step['enemy_fig'])
            value = torch.tensor([step['player_fig'] == data['winner']],dtype=serialize_in)
            to_train = redis_enc(PlayedPosition(field, mcts_probs, value))
            list_size = redisClient.llen(training_line)
            if list_size<learn_batch_size:
                redisClient.lpush(training_line, to_train)
            else:
                import madbg; madbg.set_trace()
                redisClient.lset(training_line, random.randint(0, list_size - 1), to_train)
            print('a')
        pass
    start_response('200 OK', [('Content-Type','text/html')])
    return [res.encode('utf-8')]

def model_runner():
    model = ConnNet(COLS, ROWS)
    print(f"Runner IDD {os.getpid()}")
    while True:
        _, inputs = redisClient.blmpop(0, 1, processing_line, direction='LEFT', count=batch_size)
        fields = list(map(redis_dec, inputs))
        decoded = torch.stack(list(map(lambda x: x.field ,fields)))
        policies, values = model(decoded)
        for policy, value, pos_data in zip(policies, values, fields):
            redisClient.lpush(pos_data.uuid, json.dumps({'policy': policy.tolist(), 'value': value.tolist()}))
    
    
if __name__ == 'uwsgi_file_server':
    prc = Process(target=model_runner, )
    prc.start()
    print("StArted SERVER")

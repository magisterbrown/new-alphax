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
batch_size = 16
ROWS = os.environ.get("ROWS", 3)
COLS = os.environ.get("COLS", 3)
DTYPE = torch.float32
serialize_in = DTYPE
redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)

def field_to_tenor(field: List[int], my_fig: int, enemy_fig: int) -> torch.Tensor:
    positions = np.resize(np.array(field), (ROWS, COLS))
    return torch.stack([
            torch.tensor(positions==my_fig, dtype=serialize_in),
            torch.tensor(positions==enemy_fig, dtype=serialize_in)
        ])

def application(env, start_response):
    if(env['REQUEST_METHOD'] == 'GET'):
        game = parse_qs(env['QUERY_STRING'])
        positions = np.resize(np.array(json.loads(game['field'][0])), (ROWS, COLS))
        analyzible_field = torch.stack([
            torch.tensor(positions==int(game['my_figure'][0]), dtype=serialize_in),
            torch.tensor(positions==int(game['enemy_figure'][0]), dtype=serialize_in)
        ])
        posed = DataPosition(field_to_tenor(json.loads(game['field'][0]), int(game['my_figure']), int(game['enemy_figure'])))
        redisClient.lpush(processing_line, redis_enc(posed))
        _, res = redisClient.blpop(posed.uuid)
    elif(env['REQUEST_METHOD'] == 'POST'):
        data = json.loads(env['wsgi.input'].read().decode('utf-8'))
        result = list()
        for step in data['steps']:
            probs=step['probs']
            mcts_probs = torch.zeros((COLS),dtype=serialize_in)
            mcts_probs[list(probs.keys())] = torch.tensor(scipy.special.softmax(1.0/temp * np.log(np.array(list(probs.values())) + 1e-10)),dtype=serialize_in)
            result.append
            import madbg; madbg.set_trace()
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

import os
import json
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, Pipe
from urllib.parse import parse_qs
from dataclasses import dataclass
from multiprocessing.connection import Connection
from model import ConnNet
from typing import List
from dataclasses import asdict
import redis
import pickle
import base64
import io
import uuid
import scipy
import random
import asyncio

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
learn_batch_size = 8 
batch_size = 16
ROWS = int(os.environ.get("ROWS", 3))
COLS = int(os.environ.get("COLS", 4))
DTYPE = torch.float32
serialize_in = DTYPE
redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)
temp = 1e-3

def field_to_tenor(field: List[int], my_fig: int, enemy_fig: int) -> torch.Tensor:
    positions = np.reshape(np.array(field), (ROWS, COLS))
    return torch.stack([
            torch.tensor(positions==my_fig, dtype=serialize_in),
            torch.tensor(positions==enemy_fig, dtype=serialize_in)
        ])

def application(env, start_response):
    if(env['REQUEST_METHOD'] == 'GET'):
        game = parse_qs(env['QUERY_STRING'])
        trn = field_to_tenor(json.loads(game['field'][0]), int(game['my_figure'][0]), int(game['enemy_figure'][0]))
        posed = DataPosition(trn)
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
            value = 0 if data['winner']==0 else -1 if step['player_fig'] == data['winner'] else 1
            value = torch.tensor([value], dtype=serialize_in)
            to_train = redis_enc(PlayedPosition(field, mcts_probs, value))
            list_size = redisClient.llen(training_line)
            if list_size<learn_batch_size:
                redisClient.lpush(training_line, to_train)
            else:
                redisClient.lset(training_line, random.randint(0, list_size - 1), to_train)

            res = json.dumps({'status': 'OK'})
    start_response('200 OK', [('Content-Type','text/html')])
    return [res.encode('utf-8')]

def model_runner():
    model = ConnNet(COLS, ROWS)
    print(f"Runner IDD {os.getpid()}")
    while True:
        _, inputs = redisClient.blmpop(0, 1, processing_line, direction='LEFT', count=batch_size)
        fields = list(map(redis_dec, inputs))
        decoded = torch.stack(list(map(lambda x: x.field ,fields)))
        with torch.no_grad():
            model.eval()
            policies, values = model(decoded)
        for policy, value, pos_data in zip(policies, values, fields):
            redisClient.lpush(pos_data.uuid, json.dumps({'policy': policy.tolist(), 'value': value.tolist()}))



def model_trainer():
    model = ConnNet(COLS, ROWS)
    lr = 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    while redisClient.llen(training_line) < learn_batch_size:
        asyncio.sleep(0.05)
    
    model.train()
    repets = 4
    while True:
        import madbg; madbg.set_trace()
        for i in range(repets):
            all_values = redisClient.lrange(training_line, 0, -1)
            fields, probs, values = list(map(torch.stack, zip(*map(lambda x:list(asdict(x).values()),map(redis_dec, all_values)))))
            pred_probs, pred_values = model(fields)
            loss = F.cross_entropy(pred_probs, probs)+F.mse_loss(pred_values, values)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'checkpoints/latest.pth')

        break


    pass
    
    
if __name__ == 'uwsgi_file_server':
    prc = Process(target=model_runner, )
    trn = Process(target=model_trainer, )
    prc.start()
    trn.start()
    print("StArted SERVER")

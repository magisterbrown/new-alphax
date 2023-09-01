import os
import datetime
import json
import torch
import numpy as np
import time
import torch.nn.functional as F
from torch.multiprocessing import Process, Queue, Pipe
from torch.utils.tensorboard import SummaryWriter
from urllib.parse import parse_qs
from dataclasses import dataclass
from multiprocessing.connection import Connection
from model import ConnNet
from typing import List, Dict
from dataclasses import asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import redis
import pickle
import base64
import io
import uuid
import scipy
import random
import time

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
learn_batch_size = 256
batch_size = 16
ROWS = int(os.environ.get("ROWS", 3))
COLS = int(os.environ.get("COLS", 4))
DTYPE = torch.float32
serialize_in = DTYPE
redisClient = redis.Redis(host='localhost', port=6379, decode_responses=True)
temp = 1e-3
weight_path='checkpoints/latest.pth'

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
                #TODO: get probabilities from use
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
    print('RUNNER SPAWNED')
    device = torch.device('cuda')
    writer = SummaryWriter(log_dir='board_logs')
    preds_count = 0
    model = ConnNet(COLS, ROWS)
    model.to(device)
    weights_age=0
    proc_size = 1
    report_freq=30 
    while True:
        try:
            new_age = os.path.getmtime(weight_path)
            if(weights_age+10<new_age):
                model.load_state_dict(torch.load(weight_path))
                writer.add_text('Processing/Loaded model weights at', f'Loaded weights at: {datetime.datetime.fromtimestamp(new_age)}')
        except:
            pass
        _, inputs = redisClient.blmpop(0, 1, processing_line, direction='LEFT', count=batch_size)
        fields = list(map(redis_dec, inputs))
        proc_size=proc_size*(1-1/report_freq)+len(fields)*(1/report_freq)
        if(preds_count%report_freq==0):
            writer.add_scalar('Processing/Prediction batch size',len(fields), preds_count)
            writer.flush()
        preds_count+=1
        decoded = torch.stack(list(map(lambda x: x.field ,fields)))
        with torch.no_grad():
            model.eval()
            policies, values = model(decoded.to(device))
        for policy, value, pos_data in zip(policies, values, fields):
            redisClient.lpush(pos_data.uuid, json.dumps({'policy': policy.tolist(), 'value': value.tolist()}))

def add_reads(reads: Dict[int, int], values: List[str]) -> Dict[int, int]:
    renewed = dict()
    for v in values:
        key = hash(v)
        try:
            renewed[key] = reads[key]+1
        except:
            renewed[key] = 0
    return renewed 


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


ten_num = lambda x:x.cpu().detach().item()
def model_trainer():
    device = torch.device('cuda')
    writer = SummaryWriter(log_dir='board_logs')
    fig, ax = plt.subplots()
    def report_reads(reads: dict[int, int], step: int):
        ax.clear()
        ax.bar(range(len(reads)), reads.values())
        writer.add_figure('Training/age of samples', fig, global_step=step)

    model = ConnNet(COLS, ROWS)
    model.to(device)
    
    lr = 2e-3
    lr_mult=1
    kl_targ=0.02
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_load_progress = tqdm(total=learn_batch_size, desc='loading first batch')
    while redisClient.llen(training_line) < learn_batch_size:
        batch_load_progress.n = redisClient.llen(training_line)
        writer.add_text('Processing/Loading first batch',str(batch_load_progress))
        time.sleep(0.5)
    batch_load_progress.n = learn_batch_size
    writer.add_text('Processing/Loading first batch',str(batch_load_progress))
    
    model.train()
    reads = dict()
    repeats = 40
    epochs = 5
    step = 0 
    while True:
        all_values = redisClient.lrange(training_line, 0, -1)
        reads = add_reads(reads, all_values)
        fields, probs, values = list(map(torch.stack, zip(*map(lambda x:list(asdict(x).values()),map(redis_dec, all_values)))))
        kl = kl_targ
        for i in range(epochs):
            optimizer.zero_grad()
            set_learning_rate(optimizer, lr*lr_mult)
            pred_probs, pred_values = model(fields.to(device))
            loss = F.cross_entropy(pred_probs, probs.to(device))+F.mse_loss(pred_values, values.to(device))
            loss.backward()
            optimizer.step()
            try:
                with torch.no_grad():
                    kl = torch.mean(torch.sum(old_probs*(torch.log(old_probs+1e-10)-torch.log(pred_probs+1e-10)),axis=1))
            except NameError:
                old_probs, old_values = pred_probs, pred_values 
            if kl > kl_targ*4:
                break
        del old_probs, old_values
        kl = ten_num(kl)
        writer.add_scalar('Training/Loss',ten_num(loss), step)
        writer.add_scalar('Training/Learning Rate',lr*lr_mult, step)
        writer.add_scalar('Training/KL divergence',kl, step)
        writer.add_scalar('Training/Percent of draws',ten_num((values==0).sum())/values.shape[0], step)
        writer.flush()

        if(kl > (kl_targ * 2)):
            lr_mult /= 1.5
        elif(kl < (kl_targ / 2)):
            lr_mult *= 1.5
        lr_mult=max(min(lr_mult, 10), 0.1)

        step+=1
        if(step%repeats==(repeats-1)):
            torch.save(model.state_dict(), weight_path)
            report_reads(reads, step)
    
if __name__ == 'uwsgi_file_server':
    prc = Process(target=model_runner, daemon=True)
    trn = Process(target=model_trainer, daemon=True)
    prc.start()
    trn.start()
    print("StArted SERVER")

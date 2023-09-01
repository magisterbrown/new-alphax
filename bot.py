from model_server.model import ConnNet
from kaggle_environments.utils import Struct
from kaggle_environments import make
from mcts import TreeNode, find_best_playout, tree_to_dot
import os
import torch

from typing import Callable, List, Tuple
from kaggle_environments.core import Environment
import numpy as np

def playout(observation: Struct, 
        configuration: Struct,
        root: TreeNode, 
        get_prediction: Callable[[List[int], TreeNode], Tuple[np.ndarray, float]]):
    env = env.clone()

    steps, leaf = find_best_playout(root)
    for step in steps:
        env.step([step]*2)
    if env.done:
        leaf.update(0 if env.state[0]['reward']==0 else 1) 
        return 
    field = env.state[0]['observation']['board']
    predicted_probs, value = get_prediction(field, leaf)
    columns = env.configuration['columns']
    ilegal_moves = np.nonzero(np.array(field)[:columns]!=0)[0]
    predicted_probs[ilegal_moves] = 0
    leaf.expand(predicted_probs)

    leaf.update(value)


ROWS = int(os.environ.get("ROWS", 3))
COLS = int(os.environ.get("COLS", 4))
INAROW = 3 
root = TreeNode()

def get_random_pred(field: List[int], leaf: TreeNode) -> Tuple[np.ndarray, float]:
    return np.array([0.25, 0.25, 0.25, 0.25]), 0  

def field_to_tenor(field: List[int], my_fig: int, enemy_fig: int) -> torch.Tensor:
        positions = np.reshape(np.array(field), (ROWS, COLS))
        serialize_in = torch.float32
        return torch.unsqueeze(torch.stack([
                    torch.tensor(positions==my_fig, dtype=serialize_in),
                    torch.tensor(positions==enemy_fig, dtype=serialize_in)
                ]),0 )


model = ConnNet(COLS, ROWS)  
model.eval()
def get_net_pred(field: List[int], leaf: TreeNode) -> Tuple[np.ndarray, float]:
    figures = [1,2]
    tensor = field_to_tenor(field, figures[leaf.player], figures[not leaf.player])
    with torch.no_grad():
        probs, value= model(tensor)
    return probs[0].numpy(), value.item()

    
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def alpha_bot(observation: Struct, configuration: Struct):
    assert COLS==configuration.columns, 'Wrong columns'
    assert ROWS==configuration.rows, 'Wrong rows'
    print(observation)
    print(configuration)
    print('___________________')
    step_env = make('connectx', debug=True, configuration=configuration, steps=[[{'observation':observation}, {'observation': {}}]])
    step_env.step([3,3])
    step_env.step([3,3])
    #print(step_env.state)
    root = TreeNode()

    for i in range(20):
        playout(step_env, root, get_random_pred)#get_net_pred) 
    
    probs = np.zeros(configuration.columns, np.float32)
    temp = 1e-3
    #TODO: just choose max
    visits = [v._visit_count for v in root.children.values()]
    visits = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
    step = np.random.choice(list(root.children.keys()), p=visits)
    return int(step)

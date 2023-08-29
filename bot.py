from model_server.model import ConnNet
from kaggle_environments.utils import Struct
from mcts import TreeNode, find_best_playout
import os

ROWS = int(os.environ.get("ROWS", 3))
COLS = int(os.environ.get("COLS", 4))
INAROW = 3 
#model = ConnNet(COLS, ROWS)  
def alpha_bot(observation: Struct, configuration: Struct):
    print(observation)
    print(configuration)
    print(configuration.columns)
    assert COLS==configuration.columns, 'Wrong columns'
    assert ROWS==configuration.rows, 'Wrong rows'
    return 0

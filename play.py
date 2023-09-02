from kaggle_environments import evaluate, make, utils
from kaggle_environments.core import Environment
from kaggle_environments.envs.connectx.connectx import play, is_win
from mcts import TreeNode, find_best_playout, tree_to_dot
from typing import Callable, List, Tuple
from kaggle_environments.utils import Struct, structify
from numpy.typing import NDArray
import copy
import numpy as np
import urllib3
import json
import scipy

rows = 3
columns = 4
win = 3 
rand = 0.25
get_field = lambda x: np.array(x.state[0]['observation']['board'])
disp_env = lambda x: np.reshape(get_field(x), (rows, columns))
http = urllib3.PoolManager()
server = 'http://localhost:9000'
figures = [1,2]

#def playout(env: Environment, root: TreeNode, rand: float = 0):
#    #print("--game--\n" + str(disp_env(env)))
#    env = copy.deepcopy(env)
#    steps, leaf = find_best_playout(root)
#    for step in steps:
#        env.step([step]*2)
#    if env.done:
#        #assert env.state[1-leaf.player]['reward']!=-1, 'Lost with own step whaat???'
#        leaf.update(0 if env.state[0]['reward']==0 else 1) 
#        return 
#    field = env.state[0]['observation']['board']
#    resp = http.request('GET', server, fields={"field": json.dumps(field), "my_figure": figures[leaf.player], "enemy_figure": figures[not leaf.player]})#.json()
#    resp = json.loads(resp.data.decode('utf-8'))
#    predicted_probs = np.array(resp['policy'])
#    value = resp['value'][0]
#    #TODO: remove randomenss
#    predicted_probs = predicted_probs*(1-rand) + np.random.dirichlet(np.ones(columns),size=1)[0]*rand
#    ilegal_moves = np.nonzero(np.array(field)[:columns]!=0)[0]
#    predicted_probs[ilegal_moves] = 0
#    leaf.expand(predicted_probs)
#
#    leaf.update(value)
def visits_to_probs(probs: List[int], temp: float=1e-3) -> NDArray[float]:
    return scipy.special.softmax(1.0/temp * np.log(np.array(probs) + 1e-10))

def playout(board: List[int], 
        conf: Struct,
        root: TreeNode, 
        get_prediction: Callable[[NDArray[int], TreeNode], Tuple[NDArray[float], float]]):
    board = np.array(board)
    figures = [1,2]
    curr = root.player
    steps, leaf = find_best_playout(root)
    for step in steps:
        play(board, step, figures[curr], conf)
        curr = not curr
    
    if np.all(board[:conf.columns]!=0):
        leaf.update(0)
    elif len(steps) and is_win(board, steps[-1], figures[not leaf.player], conf, has_played=True):
        leaf.update(1)
    else:  
        predicted_probs, value = get_prediction(board, leaf)
        ilegal_moves = np.nonzero(board[:conf.columns])[0]
        predicted_probs[ilegal_moves] = 0
        leaf.expand(predicted_probs)
        leaf.update(value)
        
def get_server_pred(field: NDArray[int], leaf: TreeNode) -> Tuple[NDArray[float], float]:
    figures=[1,2]
    resp = http.request('GET', server, fields={"field": json.dumps(field.tolist()), "my_figure": figures[leaf.player], "enemy_figure": figures[not leaf.player]})#.json()
    resp = json.loads(resp.data.decode('utf-8'))
    return np.array(resp['policy']), resp['value'][0]
   
def get_random_pred(field: NDArray[int], leaf: TreeNode) -> Tuple[NDArray[float], float]:
    return np.array([0.25, 0.25, 0.25, 0.25]), 0  

def play_record():
    config = structify({'rows':rows,'columns': columns,'inarow':win})
    env = make("connectx", debug=False, configuration=config)
    root = TreeNode()
    steps = list()
    while not env.done:
        board = env.state[0]['observation']['board']
        for i in range(200):
            playout(board, config, root, get_server_pred)
        #open('tree.dot', 'w').write(tree_to_dot(root))
        visits = {k: v._visit_count for k,v in root.children.items()}
        # TODO: submit visits as probailities not as counts
        steps.append({'field': board, 'probs': visits, 'player_fig': figures[root.player], 'enemy_fig': figures[not root.player]})
        visit_probs = visits_to_probs(list(visits.values()))
        visit_probs = visit_probs*(1-rand)+np.random.dirichlet(np.ones(visit_probs.shape),size=1)[0]*rand
        
        step = np.random.choice(list(visits.keys()), p=visit_probs)
        env.step([step]*2)
        root = root.children[step]
    value = 0 if env.state[0]['reward']==0 else figures[not root.player]
    data = {'steps':steps, 'winner': value}
    resp = http.request('POST', server, body=json.dumps(data))
    resp = json.loads(resp.data.decode('utf-8'))
    print(resp)



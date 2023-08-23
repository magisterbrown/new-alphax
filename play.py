from kaggle_environments import evaluate, make, utils
from kaggle_environments.core import Environment
from mcts import TreeNode, find_best_playout, tree_to_dot
import copy
import numpy as np
import urllib3
import json

rows = 3
columns = 3
win = 2 
get_field = lambda x: np.array(x.state[0]['observation']['board'])
disp_env = lambda x: np.reshape(get_field(x), (rows, columns))
http = urllib3.PoolManager()
server = 'http://localhost:9000'
figures = [1,2]

def playout(env: Environment, root: TreeNode, rand: float = 0):
    #print("--game--\n" + str(disp_env(env)))
    env = copy.deepcopy(env)
    steps, leaf = find_best_playout(root)
    for step in steps:
        env.step([step]*2)
    if env.done:
        #assert env.state[1-leaf.player]['reward']!=-1, 'Lost with own step whaat???'
        leaf.update(0 if env.state[0]['reward']==0 else 1) 
        print('Played')
        return 
    field = env.state[0]['observation']['board']
    resp = http.request('GET', server, fields={"field": json.dumps(field), "my_figure": figures[leaf.player], "enemy_figure": figures[not leaf.player]}).json()
    predicted_probs = np.array(resp['policy'])
    value = resp['value'][0]
    predicted_probs = predicted_probs*(1-rand) + np.random.dirichlet(np.ones(columns),size=1)[0]*rand
    ilegal_moves = np.nonzero(np.array(field)[:columns]!=0)[0]
    predicted_probs[ilegal_moves] = 0
    leaf.expand(predicted_probs)

    leaf.update(value)

def play():
    env = make("connectx", debug=False, configuration={'rows':rows,'columns': columns,'inarow':win})
    root = TreeNode()
    while not env.done:
        for i in range(60):
            playout(env, root)
        dot = tree_to_dot(root)
        with open('tree.dot', 'w') as f:
            f.write(dot)

        step = 1#int(np.random.choice(legal_moves))
        print(type(env))
        env.step([step]*2)
        break



play()
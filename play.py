from kaggle_environments import evaluate, make, utils
from kaggle_environments.core import Environment
from mcts import TreeNode, find_best_playout, tree_to_dot
import copy
import numpy as np
import urllib3
import json

rows = 3
columns = 4
win = 3 
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
        return 
    field = env.state[0]['observation']['board']
    resp = http.request('GET', server, fields={"field": json.dumps(field), "my_figure": figures[leaf.player], "enemy_figure": figures[not leaf.player]})#.json()
    resp = json.loads(resp.data.decode('utf-8'))
    predicted_probs = np.array(resp['policy'])
    value = resp['value'][0]
    #TODO: remove randomenss
    predicted_probs = predicted_probs*(1-rand) + np.random.dirichlet(np.ones(columns),size=1)[0]*rand
    ilegal_moves = np.nonzero(np.array(field)[:columns]!=0)[0]
    predicted_probs[ilegal_moves] = 0
    leaf.expand(predicted_probs)

    leaf.update(value)

def play():
    env = make("connectx", debug=False, configuration={'rows':rows,'columns': columns,'inarow':win})
    root = TreeNode()
    steps = list()
    while not env.done:
        for i in range(120):
            playout(env, root, rand=0.25)
        #open('tree.dot', 'w').write(tree_to_dot(root))
        visits = {k: v._visit_count for k,v in root.children.items()}
        # TODO: submit visits as probailities not as counts
        steps.append({'field':env.state[0]['observation']['board'], 'probs': visits, 'player_fig': figures[root.player], 'enemy_fig': figures[not root.player]})
        #And randomness to probability mentioned previously and choose using it
        step = max(visits, key=visits.get)#int(np.random.choice(legal_moves))
        env.step([step]*2)
        root = root.children[step]
    value = 0 if env.state[0]['reward']==0 else figures[not root.player]
    data = {'steps':steps, 'winner': value}
    resp = http.request('POST', server, body=json.dumps(data))
    resp = json.loads(resp.data.decode('utf-8'))
    print(resp)




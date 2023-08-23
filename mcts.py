from __future__ import annotations
from typing import Optional, Tuple, List, Union, Dict 
from graphviz.graphs import Digraph
import graphviz


import numpy as np

class TreeNode:
    Q: float = 0 #Action value
    _visit_count: int = 0 #Visit count

    def __init__(self, parent: Optional[TreeNode] = None, prior: float = 0, player: bool = False):
        self.parent = parent
        self.children = dict()
        self._P = prior #Prior probability
        self.player = player

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_next(self) -> Tuple[int, TreeNode]:
        assert not self.is_leaf(), "Trying to continue not expanded or finished game."
        next_move = max(self.children, key=self.children.get) 
        return next_move, self.children[next_move]

    def expand(self, probs: List[float]):
        self.children = {ind: TreeNode(self, prob, not self.player) for ind, prob in enumerate(probs) if prob>0}

    def update(self, value: int):
        self._visit_count+=1
        self.Q+=(value - self.Q) / self._visit_count
        if self.parent is not None:
            self.parent.update(-1*value)

    def forget_parent(self):
        self.parent = None

    def _get_value(self) -> float:
        parent_popularity = 1
        if self.parent is not None:
            parent_popularity = np.sqrt(self.parent._visit_count)
        return self.Q+(parent_popularity*self._P)/(1+self._visit_count)

    def __lt__(self, other: TreeNode) -> bool:
        return self._get_value() < other._get_value()

def find_best_playout(root: TreeNode) -> Union[List[int], TreeNode]:
    root.forget_parent()
    steps = list()
    while(not root.is_leaf()):
        step, root = root.get_next()
        steps.append(step)
    return steps, root

#TODO: Rewrite with a normal tree and end nodes
def tree_to_dot(root: TreeNode) -> str:
    dot = graphviz.Digraph()
    def node_numerer() -> str:
        ct = 0
        while True:
            yield str(ct)
            ct+=1
    numerer = iter(node_numerer())
    norm_color = lambda x: str(int(x*5+6))
    node_colors = ['darkgoldenrod2', 'darkolivegreen2'] 
    dot.attr('edge', colorscheme='rdylgn11')
    dot.attr(splines='line')
    def recursive_call(parent: str, node: TreeNode, graph: Digraph):
        node_n = next(numerer)
        dot.node(node_n, str(node.player+1), fillcolor=node_colors[node.player], style='filled')
        weight = node._get_value()
        dot.edge(parent, node_n, weight=str(weight), color=str(min(int(weight*5+6),11)), label='%.2f'%weight )
        for idx, child in node.children.items():
            recursive_call(node_n, child, dot)
    node_n = next(numerer)
    dot.node(node_n, str(root.player+1), fillcolor=node_colors[root.player], style='filled')
    for idx, child in root.children.items():
        recursive_call(node_n, child, dot)
    
    return dot.source


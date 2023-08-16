from __future__ import annotations
from typing import Optional, Tuple, List, Union 


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
        assert not self.is_leaf() | "Trying to continue not expanded or finished game."
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
        return self._get_value() < other.get_value()

def find_best_playout(root: TreeNode) -> Union[List[int], TreeNode]:
    root.forget_parent()
    steps = list()
    while(not root.is_leaf()):
        step, root = root.get_next()
        setps.append(step)
    return steps, root


from mcts import MCTS, TreeNode
import unittest

class TestTreeNode(unittest.TestCase):

    def test_node(self):
        node = TreeNode()
        print(node)

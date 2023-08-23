from mcts import TreeNode, tree_to_dot
from unittest import skip
import unittest

class TestTreeNode(unittest.TestCase):

    def setUp(self):
        self.node = TreeNode()
        probs = [0, 0.2, 0.3, 0 ,0.5]
        probs1 = [0.5, 0.5]
        probs2 = [0.9,0, 0.1]
        self.node.expand(probs)
        self.node.children[2].expand(probs1)
        self.node.children[4].expand(probs2)
    
    @skip
    def test_node(self):
        self.assertEqual(len(self.node.children),3)

    def test_dot(self):
        self.node.children[2].children[1].update(1)
        self.node.children[2].children[1].update(1)
        self.node.children[4].children[2].update(0)
    
        dot = tree_to_dot(self.node)
        print(dot)
        with open('tree.dot', 'w') as f:
            f.write(dot)

if __name__=='__main__':
    unittest.main()

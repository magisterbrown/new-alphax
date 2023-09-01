from unittest import skip
from kaggle_environments import evaluate, make, utils
from bot import alpha_bot
import unittest


class TestBot(unittest.TestCase):
    configuration={'rows':3,'columns': 4,'inarow':3}

    def test_random(self):
        res = evaluate("connectx", [alpha_bot, "random"], num_episodes=1, debug=True, configuration=self.configuration)
        print(res)
    @skip 
    def test_right(self):
        res = evaluate("connectx", ["random", alpha_bot], num_episodes=1, debug=True, configuration=self.configuration)
    
    @skip
    def test_both(self):
        res = evaluate("connectx", [alpha_bot, alpha_bot], num_episodes=1, debug=True, configuration=self.configuration)
if __name__=='__main__':

    unittest.main()

from unittest import skip
from kaggle_environments import evaluate, make, utils
from bot import alpha_bot
import unittest


class TestBot(unittest.TestCase):
    def test_random(self):
        configuration={'rows':3,'columns': 4,'inarow':3}
        res = evaluate("connectx", [alpha_bot, "random"], num_episodes=1, debug=True, configuration=configuration)
        print(res)

if __name__=='__main__':
    unittest.main()

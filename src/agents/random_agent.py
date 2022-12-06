import random
import numpy as np


# from src.environments.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified


class RandomPlayer:
    def __init__(self, args):
        self.args = args

    def act(self, state):
        # TODO how to do if-else only once?
        if self.args.m == 'coinjump':
            action = self.coinjump_actor()
        elif self.args.m == 'bigfish':
            action = self.bigfish_actor()
        elif self.args.m == 'heist':
            action = self.heist_actor()
        return action

    def coinjump_actor(self):
        # action = coin_jump_actions_from_unified(random.randint(0, 10))
        action = random.randint(0, 10)
        return action

    def bigfish_actor(self):
        action = np.random.randint([9])
        return action

    def heist_actor(self):
        action = np.random.randint([9])
        return action

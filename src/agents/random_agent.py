import random
import numpy as np


class RandomPlayer:
    def __init__(self, args):
        self.args = args

    def act(self, state):
        # TODO how to do if-else only once?
        if self.args.m == 'getout':
            action = self.coinjump_actor()
        elif self.args.m == 'threefish':
            action = self.threefish_actor()
        elif self.args.m == 'loot':
            action = self.loot_actor()
        return action

    def coinjump_actor(self):
        # action = coin_jump_actions_from_unified(random.randint(0, 10))
        return random.randint(0, 10)

    def threefish_actor(self):
        return np.random.randint([9])

    def loot_actor(self):
        return np.random.randint([9])

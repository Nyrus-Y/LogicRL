import random

import gym
import numpy as np
from gym import spaces

from src.environments.coinjump.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS, coin_jump_actions_from_unified
from src.environments.coinjump.coinjump.coinjump.helpers import create_coinjump_instance
from src.environments.coinjump.utils_cj import extract_for_explaining

class CoinJumpEnvLogic(gym.Env):
    """
        Coinjump:include enemy, key and door.
        trained with NSFRReasoner and mlp
    """
    # metadata = {'render.modes': ['human']}
    metadata = {'render.modes': []}

    def __init__(self, generator_args=None, **kwargs):
        self.rng = random.Random()
        self._generator_args = generator_args
        self._kwargs = kwargs
        self.coinjump = None
        self.create_new_coinjump()
        self.action_space = spaces.Discrete(CJA_NUM_EXPLICIT_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=50, shape=(60,), dtype=np.float)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        # self._take_action(action)
        reward = self.coinjump.step(coin_jump_actions_from_unified(action))
        # reward = self._get_reward()
        ob = self.observe_state()
        episode_over = self.coinjump.level.terminated
        return ob, reward, episode_over, {}

    def observe_state(self):

        ob = extract_for_explaining(self.coinjump)

        return ob

    def create_new_coinjump(self):
        self.coinjump = create_coinjump_instance(V1=True,
            seed=self.rng.randint(0, 1000000000), print_seed=False, render=False,
            resource_path=None, start_on_first_action=False,
            generator_args=self._generator_args, **self._kwargs)

    def reset(self):
        self.create_new_coinjump()
        return self.observe_state()

    def render(self, mode='human', close=False):
        if close:
            return
        raise NotImplementedError("")

    # def _take_action(self, action):
    #    self.coinjump1.step(action)
    def get_coinjump(self):
        return self.coinjump

    def _get_reward(self):
        return self.coinjump.level.get_reward()

    def _seed(self, seed):
        self.rng = random.Random(seed)


gym.envs.register(
    id='CoinJumpEnvLogic-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.CoinJumpEnvLogic:CoinJumpEnvLogic',
    max_episode_steps=300,
)

import random

import gym
import numpy as np
from gym import spaces

from src.CoinJump.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS, coin_jump_actions_from_unified
from src.CoinJump.coinjump.coinjump.helpers import create_coinjump_instance
from src.util import extract_for_explaining
from src.CoinJump.coinjump_learn.training.data_transform import extract_state, sample_to_model_input_KD


class CoinJumpEnvKD(gym.Env):

    #metadata = {'render.modes': ['human']}
    metadata = {'render.modes': []}

    def __init__(self, generator_args=None, **kwargs):
        self.rng = random.Random()
        self._generator_args = generator_args
        self._kwargs = kwargs
        self.coinjump = None
        self.create_new_coinjump()

        self.action_space = spaces.Discrete(CJA_NUM_EXPLICIT_ACTIONS)
        # Example for using image as input:

        self.observation_space = spaces.Box(low=0, high=50, shape=(60,), dtype=np.float),

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
        #self._take_action(action)
        reward = self.coinjump.step(coin_jump_actions_from_unified(action))
        #reward = self._get_reward()
        ob = self.observe_state()
        episode_over = self.coinjump.level.terminated
        return ob, reward, episode_over, {}

    def observe_state(self):
        # transform to model input with no action specified
        # ob = sample_to_model_input_KD((extract_state(self.coinjump), None), no_dict=True)
        ob = extract_for_explaining(self.coinjump)
        #if no_dict=False: we get a dict with ob['base'] and ob['entities'] entries
        return ob

    def create_new_coinjump(self):
        self.coinjump = create_coinjump_instance(key_door=True,
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

    #def _take_action(self, action):
    #    self.coinjump1.step(action)

    def _get_reward(self):
        return self.coinjump.level.get_reward()

    def _seed(self, seed):
        self.rng = random.Random(seed)


gym.envs.register(
     id='CoinJumpEnvKD-v0',
     entry_point='src.CoinJump.coinjump_learn.env.coinJumpEnvKD:CoinJumpEnvKD',
     max_episode_steps=300,
)
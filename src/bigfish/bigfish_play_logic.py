import random
import sys
import gym3
import numpy as np

from utils_procgen import InteractiveEnv
from procgen import ProcgenGym3Env
from src.utils_bf import extract_reasoning_state, explaining_nsfr, action_select

INTERACTIVE = False

env_name = "bigfishm"

env = ProcgenGym3Env(num=1, env_name=env_name, render_mode="rgb_array")
env = gym3.ViewerWrapper(env, info_key="rgb")

step = 1
NB_DONE = 0
TO_SUCCEED = 1

total_reward = 0

# [
#     ("LEFT",),
#     ("DOWN",),
#     (),
#     ("UP",),
#     ("RIGHT",),
# ]
action_space = [1, 3, 4, 5, 7]

rew, obs, done = env.observe()
extracted_reasoning_states = extract_reasoning_state(obs['positions'])
prednames = ['up_to_eat', 'left_to_eat', 'down_to_eat', 'right_to_eat',
             'up_to_dodge', 'down_to_dodge']

while NB_DONE < TO_SUCCEED:
    explaining = explaining_nsfr(extracted_reasoning_states, 'bigfish_simplified_actions', prednames)
    action = action_select(explaining)
    env.act(action)

    rew, obs, done = env.observe()

    print(explaining)
    extracted_reasoning_states = extract_reasoning_state(obs["positions"])

    # print(f"reward : {rew}")

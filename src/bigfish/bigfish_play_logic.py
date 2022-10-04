import random
import sys
import gym3
import numpy as np

from procgen import ProcgenGym3Env
from src.utils_bf import extract_reasoning_state, nsfr, explain, action_select

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
# action_space = [1, 3, 4, 5, 7]

rew, obs, done = env.observe()
extracted_reasoning_states = extract_reasoning_state(obs['positions'])

explaining_env = 'bigfish_simplified_actions'
NSFR = nsfr(explaining_env)

while NB_DONE < TO_SUCCEED:
    explaining = explain(NSFR, extracted_reasoning_states)
    action = action_select(explaining)
    env.act(action)

    rew, obs, done = env.observe()

    print(explaining)
    extracted_reasoning_states = extract_reasoning_state(obs["positions"])

    # print(f"reward : {rew}")

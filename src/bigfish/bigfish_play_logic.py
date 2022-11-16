import random
import sys
import gym3
import numpy as np

from src.bigfish.procgen import ProcgenGym3Env
from src.utils_bf import extract_reasoning_state, nsfr, explain, action_select

INTERACTIVE = False

env_name = "bigfishc"

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
extracted_reasoning_states = extract_reasoning_state(obs['positions'], env='size')

# explaining_env = 'color_bigfish'
explaining_env = 'bigfish_simplified_actions'
NSFR = nsfr(explaining_env)
last_explaining = ""
while NB_DONE < TO_SUCCEED:
    explaining = explain(NSFR, extracted_reasoning_states)
    action = action_select(explaining)
    env.act(action)

    rew, obs, done = env.observe()

    # print(explaining)
    if last_explaining != explaining.head.pred.name:
        print(explaining.head.pred.name)
        last_explaining = explaining.head.pred.name

    extracted_reasoning_states = extract_reasoning_state(obs["positions"], env="size")
    if done:
        print("--------------------------new game--------------------------")
    # print(f"reward : {rew}")

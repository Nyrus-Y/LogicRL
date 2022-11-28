import random
from argparse import ArgumentParser
import pathlib
import pickle
import torch
import numpy as np
import os
import gym3

from procgen import ProcgenGym3Env
from src.utils_bf import extract_state,simplify_action
from ppo_bigfish import ActorCritic


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play coinjump")
    parser.add_argument("-m", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", type=int)
    args = parser.parse_args()

    # TODO change path of model
    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'ppo_bigfish_model', model_name)
        # model_file = f"../src/ppo_coinjump_model/{input('Enter file name: ')}"

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, set_eval=True):
    with open(model_path, "rb") as f:
        model = torch.load(f)

    if isinstance(model, ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def run():
    args, model_file = parse_args()

    model = load_model(model_file)

    env_name = "bigfishm"
    env = ProcgenGym3Env(num=1, env_name=env_name, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    reward, obs, done = env.observe()
    state = extract_state(obs['positions'])
    ep_reward = 0
    while True:

        prediction = model(torch.tensor(state, device='cuda:0'))
        action = torch.argmax(prediction).cpu().numpy().reshape(-1)
        action = simplify_action(action)
        # print(action)
        env.act(action)
        rew, obs, done = env.observe()
        print(action)
        # if action[0] == 4:
        #     rew += 0.001
        # ep_reward += rew
        # print(rew)
        state = extract_state(obs['positions'])
        if done:
            ep_reward = 0


if __name__ == "__main__":
    run()

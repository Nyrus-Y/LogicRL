import random
import time
from argparse import ArgumentParser
import pathlib
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import gym3

from src.utils_bf import extract_reasoning_state, num_action_select, print_explaining
from src.bigfish.training.mlpCriticController import MLPController
from src.valuation_bf import BFValuationModule
from src.nsfr.facts_converter import FactsConverter
from src.logic_utils import build_infer_module, get_lang
from src.nsfr_bf import NSFReasoner
from procgen import ProcgenGym3Env
from src.bigfish.actorcritic_bigfish_logic_policy import NSFR_ActorCritic

# from src.bigfish.closeby import NSFR_ActorCritic
KEY_r = 114


#
#
# class NSFR_ActorCritic(nn.Module):
#     def __init__(self):
#         super(NSFR_ActorCritic, self).__init__()
#
#         self.actor = self.get_nsfr_model(train=False)
#         self.critic = MLPController(out_size=1)
#
#     def forward(self):
#         raise NotImplementedError
#
#     def act(self):
#         pass
#
#     def get_nsfr_model(self, train=True):
#         # current_path = os.getcwd()
#         # lark_path = os.path.join(current_path, '..', 'lark/exp.lark')
#         # lang_base_path = os.path.join(current_path, '..', 'data/lang/')
#
#         lark_path = 'lark/exp.lark'
#         lang_base_path = 'data/lang/'
#         device = torch.device('cuda:0')
#         lang, clauses, bk, atoms = get_lang(
#             lark_path, lang_base_path, 'bigfish', 'bigfish_simplified_actions')
#
#         VM = BFValuationModule(lang=lang, device=device)
#         FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
#         m = len(clauses)
#         IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
#         # Neuro-Symbolic Forward Reasoner
#         NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=True)
#         return NSFR


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play bigfish")
    parser.add_argument("-m", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", type=int)
    args = parser.parse_args()

    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'nsfr_bigfish_model', model_name)

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, set_eval=True):
    with open(model_path, "rb") as f:
        model = NSFR_ActorCritic()
        # a = model.state_dict()
        # model_1 = torch.load(f)
        model.load_state_dict(state_dict=torch.load(f))

    if isinstance(model, NSFR_ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def run():
    args, model_file = parse_args()

    model = load_model(model_file)

    seed = random.seed() if args.seed is None else int(args.seed)

    env_name = "bigfishm"
    env = ProcgenGym3Env(num=1, env_name=env_name, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    reward, obs, done = env.observe()
    state = extract_reasoning_state(obs['positions'])

    print(model.state_dict())
    while True:
        # select action with policy
        actions = model(state)
        print_explaining(actions)
        action = num_action_select(actions, trained=True)
        # state, reward, done, _ = env.step(action)
        env.act(action)
        rew, obs, done = env.observe()
        state = extract_reasoning_state(obs["positions"])


if __name__ == "__main__":
    run()

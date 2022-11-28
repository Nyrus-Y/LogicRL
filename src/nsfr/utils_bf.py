import numpy
import numpy as np
import torch
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
# from .valuation_bf import BFValuationModule
from .facts_converter import FactsConverter
from .nsfr_bf import NSFReasoner
from .logic_utils import build_infer_module, get_lang, get_prednames


device = torch.device('cuda:0')


def fuzzy_position(pos1, pos2, keyword):
    x = pos2[:, 0] - pos1[:, 0]
    y = pos2[:, 1] - pos1[:, 1]
    tan = torch.atan2(y, x)
    degree = tan[:] / torch.pi * 180

    # if keyword == 'top':
    #     probs = 1 - abs(degree[:] - 90) / 45
    #     result = torch.where((135 >= degree) & (degree >= 45), probs, 0)
    # elif keyword == 'top_left':
    #     probs = 1 - abs(degree[:] - 135) / 45
    #     result = torch.where((180 >= degree) & (degree >= 90), probs, 0)
    # elif keyword == 'left':
    #     probs = 1 - abs(degree[:] - 180) / 45
    #     result = torch.where((degree <= -135) & (degree >= 135), probs, 0)
    # elif keyword == 'bottom_left':
    #     probs = 1 - abs(degree[:] + 135) / 45
    #     result = torch.where((-90 >= degree) & (degree >= -180), probs, 0)
    # elif keyword == 'bottom':
    #     probs = 1 - abs(degree[:] + 90) / 45
    #     result = torch.where((-45 >= degree) & (degree >= -135), probs, 0)
    # elif keyword == 'bottom_right':
    #     probs = 1 - abs(degree[:] + 45) / 45
    #     result = torch.where((0 >= degree) & (degree >= -90), probs, 0)
    # elif keyword == 'right':
    #     probs = 1 - abs(degree[:]) / 45
    #     result = torch.where((45 >= degree) & (degree >= -45), probs, 0)
    # elif keyword == 'top_right':
    #     probs = 1 - abs(degree[:] - 45) / 45
    #     result = torch.where((90 >= degree) & (degree >= 0), probs, 0)

    if keyword == 'top':
        probs = 1 - abs(degree[:] - 90) / 90
        result = torch.where((180 >= degree) & (degree >= 0), probs, 0)
    elif keyword == 'left':
        probs = (abs(degree[:]) - 90) / 90
        result = torch.where((degree <= -90) | (degree >= 90), probs, 0)
    elif keyword == 'bottom':
        probs = 1 - abs(degree[:] + 90) / 90
        result = torch.where((0 >= degree) & (degree >= -180), probs, 0)
    elif keyword == 'right':
        probs = 1 - abs(degree[:]) / 90
        result = torch.where((90 >= degree) & (degree >= -90), probs, 0)

    return result


# def reward_shaping(old_state, new_state, reward):
#     old_dis = abs(old_state[0] - old_state[3]) + abs(old_state[1] - old_state[4])
#     new_dis = abs(new_state[0] - new_state[3]) + abs(new_state[1] - new_state[4])
#     if (new_dis - old_dis) < 0:
#         reward += 0.01
#     else:
#         reward -= 0.01
#     return reward


def simplify_action(action):
    """[
        ("LEFT", "DOWN"),
        ("LEFT",),
        ("LEFT", "UP"),
        ("DOWN",),
        (),
        ("UP",),
        ("RIGHT", "DOWN"),
        ("RIGHT",),
        ("RIGHT", "UP")
    ]"""
    #              [0, 1, 2, 3, 4]
    action_space = [1, 3, 4, 5, 7]
    ac_index = action[0].astype(int)
    action = action_space[ac_index]
    return np.array([action])



# def get_nsfr_model(lang, clauses, atoms, bk, device):
#     VM = BFValuationModule(lang=lang, device=device)
#     FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
#     IM = build_infer_module(clauses, atoms, lang,
#                             m=len(clauses), infer_step=2, train=False, device=device)
#     prednames = get_prednames(clauses)
#     # prednames = ['up_to_eat', 'left_to_eat', 'down_to_eat', 'right_to_eat', 'up_to_dodge', 'down_to_dodge',
#     #              'up_redundant', 'down_redundant']
#     # Neuro-Symbolic Forward Reasoner
#     NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, prednames=prednames)
#     return NSFR


# def nsfr(env):
#     current_path = os.getcwd()
#     lark_path = os.path.join(current_path, 'lark/exp.lark')
#     lang_base_path = os.path.join(current_path, 'data/lang/')
#
#     device = torch.device('cuda:0')
#     lang, clauses, bk, atoms = get_lang(
#         lark_path, lang_base_path, 'bigfish', env)
#     NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)
#     return NSFR


def explain(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T, prednames=prednames)
    explaining = NSFR.print_explaining(predicts)
    # explaining = explaining.head.pred.name
    return explaining

def action_select(explaining):
    """[
        ("LEFT",),
        ("DOWN",),
        (),
        ("UP",),
        ("RIGHT",),
    ]
    action_space = [1, 3, 4, 5, 7]

    """

    full_name = explaining.head.pred.name

    if 'left' in full_name:
        action = np.array([1])
    elif 'right' in full_name:
        action = np.array([7])
    elif 'up' in full_name:
        action = np.array([5])
    elif 'down' in full_name:
        action = np.array([3])
    return action


def num_action_select(action, trained=False):
    """
    prednames:  [
                'up_to_eat',
                'left_to_eat',
                'down_to_eat',
                'right_to_eat',
                'up_to_dodge',
                'down_to_dodge',
                'up_redundant',
                'down_redundant'
                'left_redundant',
                'right_redundant',
                'idle_redundant'
                ]

    env_actions
                [
                    ("LEFT",),
                    ("DOWN",),
                    (),
                    ("UP",),
                    ("RIGHT",),
                ]
    action_space = [1, 3, 4, 5, 7]
    """
    if trained == True:
        action = torch.argmax(action)
    # up
    if action in [0, 4, 6]:
        return np.array([5])
    # left
    elif action in [1, 8]:
        return np.array([1])
    # down
    elif action in [2, 5, 7]:
        return np.array([3])
    # right
    elif action in [3, 9]:
        return np.array([7])
    # idle
    else:
        return np.array([4])




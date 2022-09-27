import numpy
import numpy as np
import torch
import math
import os

from src.facts_converter import FactsConverter
from src.nsfr_bf import NSFReasoner
from src.logic_utils import build_infer_module, get_lang
from src.valuation_bf import BFValuationModule

device = torch.device('cuda:0')


def extract_state(obs):
    state = obs.reshape(-1)
    # return torch.tensor(state, device='cuda:0')
    return state.tolist()


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
    action_space = [1, 3, 4, 5, 7]
    ac_index = action[0].astype(int)
    action = action_space[ac_index]
    return np.array([action])


def extract_reasoning_state(states):
    """
    states = [x, y, radius]
    extracted_states = [agent, fish, r, x, y]
    size: 3*5 3 fish, 5 features
    """
    states = torch.from_numpy(states).squeeze()
    extracted_state = torch.zeros((3, 5))
    for i, state in enumerate(states):
        if i == 0:
            extracted_state[i, 0] = 1  # agent
            extracted_state[i, 2] = states[i, 2]  # radius
            extracted_state[i, 3] = states[i, 0]  # X
            extracted_state[i, 4] = states[i, 1]  # Y
        else:
            extracted_state[i, 1] = 1  # agent
            extracted_state[i, 2] = states[i, 2]  # radius
            extracted_state[i, 3] = states[i, 0]  # X
            extracted_state[i, 4] = states[i, 1]  # Y

    extracted_state = extracted_state.unsqueeze(0)
    return extracted_state.cuda()


def get_nsfr_model(lang, clauses, atoms, bk, device):
    VM = BFValuationModule(
        lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, train=False, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def explaining_nsfr(extracted_states, env, prednames):
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'bigfish', env)
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)

    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(v=V_T, prednames=prednames)

    explaining = NSFR.print_explaining(predicts)

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

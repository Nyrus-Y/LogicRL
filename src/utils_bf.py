import numpy
import numpy as np
import torch
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.facts_converter import FactsConverter
from src.nsfr_bf import NSFReasoner
from src.logic_utils import build_infer_module, get_lang, get_prednames
from src import valuation_bf

device = torch.device('cuda:0')


def extract_state(obs, train=False):
    state = obs.reshape(-1)
    # return torch.tensor(state, device='cuda:0')
    state = state.tolist()
    if train:
        return torch.tensor(state)
    else:
        return state


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
            extracted_state[i, 1] = 1  # fish
            extracted_state[i, 2] = states[i, 2]  # radius
            extracted_state[i, 3] = states[i, 0]  # X
            extracted_state[i, 4] = states[i, 1]  # Y

    extracted_state = extracted_state.unsqueeze(0)
    return extracted_state.cuda()


def get_nsfr_model(lang, clauses, atoms, bk, device):
    VM = valuation_bf.BFValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, train=False, device=device)
    prednames = get_prednames(clauses)
    # prednames = ['up_to_eat', 'left_to_eat', 'down_to_eat', 'right_to_eat', 'up_to_dodge', 'down_to_dodge',
    #              'up_redundant', 'down_redundant']
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, prednames=prednames)
    return NSFR


def nsfr(env):
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'bigfish', env)
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)
    return NSFR


def explain(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T, prednames=prednames)
    explaining = NSFR.print_explaining(predicts)

    return explaining


def print_explaining(actions):
    action = torch.argmax(actions)
    prednames = ['up_to_eat', 'left_to_eat', 'down_to_eat', 'right_to_eat', 'up_to_dodge', 'down_to_dodge',
                 'up_redundant', 'down_redundant','left_redundant','right_redundant','idle_redundant']
    return print(prednames[action])


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


def plot_weights(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(15, 5))
    plt.ylim([0, 1])
    x_label = ['up_eat', 'left_eat', 'down_eat', 'right_eat',
               'up_dodge', 'down_dodge', 'up_re', 'down_re', 'left_re', 'right_re',
               'idle_re']
    # x_label = ['Jump', 'Left_k', 'Right_k', 'Left_d',
    #            'Right_d', 'Stay', 'Jump_d', 'Left_n', 'Right_e',
    #            'Stay_n']
    x = np.arange(len(x_label)) / 2
    # width = 0.15
    # X = x - width * 2

    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()

        # X = X + width
        # plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
        plt.bar(x, W_, width=0.25, alpha=1, label='C' + str(i))
        # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))

    plt.xticks(x, x_label, fontproperties="Microsoft YaHei", size=12)
    plt.ylabel('Weights', size=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

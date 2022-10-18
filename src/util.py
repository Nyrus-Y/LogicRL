import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random

from src.facts_converter import FactsConverter
from src.nsfr import NSFReasoner
from src.logic_utils import build_infer_module, get_lang
from src.valuation import RLValuationModule, RLValuationModule_D, RLValuationModule_KD
from src.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified, CoinJumpActions

device = torch.device('cpu')


def get_nsfr_model(lang, clauses, atoms, bk, device):
    VM = RLValuationModule(
        lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, train=True, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


#
# def get_2_nsfr_model(lang, clauses, atoms, bk, device):
#     PM = YOLOPerceptionModule(e=4, d=11, device=device)
#     VM_D = RLValuationModule_D(
#         lang=lang, device=device)
#     VM_KD = RLValuationModule_KD(
#         lang=lang, device=device
#     )
#
#     def combine(VM_D, VM_KD):
#         # TODO
#         for module in VM_KD.layers:
#             if module not in VM_D.layers:
#                 print(module == VM_D.layers[0])
#                 VM_D.layers.append(module)
#
#         return VM_D
#
#     VM = combine(VM_D, VM_KD)
#     FC = FactsConverter(lang=lang, perception_module=PM,
#                         valuation_module=VM, device=device)
#     IM = build_infer_module(clauses, atoms, lang,
#                             m=len(clauses), infer_step=2, device=device)
#     # Neuro-Symbolic Forward Reasoner
#     NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
#                        infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
#
#     return NSFR

def get_nsfr(env_name):
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'coinjump', env_name)
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)
    return NSFR


def explaining_nsfr(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    # prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T)
    explaining = NSFR.print_explaining(predicts)
    return explaining


def get_predictions(extracted_states, NSFR, prednames):
    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(v=V_T, prednames=prednames)

    return predicts


#
# def explaining_nsfr_combine(extracted_states, env1, env2):
#     lark_path = '../src/lark/exp.lark'
#     lang_base_path = '../data/lang/'
#
#     device = torch.device('cuda:0')
#
#     def combine(data1, data2):
#         lang = data1['lang']
#         for pred in data2['lang'].preds:
#             if pred not in data1['lang'].preds:
#                 lang.preds.append(pred)
#         clauses = data1['clauses']
#         for clause in data2['clauses']:
#             if clause not in data1['clauses']:
#                 clauses.append(clause)
#         bks = data1['bk']
#         for bk in data2['bk']:
#             if bk not in data1['bk']:
#                 bks.append(bk)
#         atoms = data1['atoms']
#         for atom in data2['atoms']:
#             if atom not in data1['bk']:
#                 atoms.append(atom)
#         return lang, clauses, bks, atoms
#
#     lang1, clauses1, bk1, atoms1 = get_lang(
#         lark_path, lang_base_path, env1, 'coinjump')
#     lang2, clauses2, bk2, atoms2 = get_lang(
#         lark_path, lang_base_path, env2, 'coinjump')
#     data1 = {'lang': lang1, 'clauses': clauses1, 'bk': bk1, 'atoms': atoms1}
#     data2 = {'lang': lang2, 'clauses': clauses2, 'bk': bk2, 'atoms': atoms2}
#     lang, clauses, bk, atoms = combine(data1, data2)
#     NSFR = get_2_nsfr_model(lang, clauses, atoms, bk, device)
#
#     V_T = NSFR(extracted_states)
#     predicts = NSFR.predict_multi(
#         v=V_T, prednames=['left_go_get_key', 'right_go_get_key', 'left_go_to_door',
#                           'right_go_to_door'])
#
#     # predicts = NSFR.predict_multi(
#     #     v=V_T, prednames=['jump', 'left', 'right'])
#
#     explaining = NSFR.print_explaining(predicts)
#
#     return explaining


def action_select(explaining):
    """
    CJA_NOOP: Final[int] = 0
    CJA_MOVE_LEFT: Final[int] = 1
    CJA_MOVE_RIGHT: Final[int] = 2
    CJA_MOVE_UP: Final[int] = 3
    CJA_MOVE_DOWN: Final[int] = 4
    CJA_MOVE_LEFT_UP: Final[int] = 5
    CJA_MOVE_RIGHT_UP: Final[int] = 6
    CJA_MOVE_LEFT_DOWN: Final[int] = 7
    CJA_MOVE_RIGHT_DOWN: Final[int]= 8
    CJA_NUM_EXPLICIT_ACTIONS = 9
    """
    action = CoinJumpActions.NOOP

    full_name = explaining.head.pred.name

    if 'left' in full_name:
        action = coin_jump_actions_from_unified(1)
    elif 'right' in full_name:
        action = coin_jump_actions_from_unified(2)
    elif 'jump' in full_name:
        action = coin_jump_actions_from_unified(3)
    elif 'stay' in full_name:
        action = coin_jump_actions_from_unified(0)
    return action


def num_action_select(action, KD=False, V1=False, V2=False, Dodge=False):
    """
    0:jump
    1:left_go_get_key
    2:right_go_get_key
    3:left_go_to_door
    4:right_go_to_door
    5:stay
    6:jump_over_door
    7:left_for_nothing
    8:right_go_to_enemy
    9:stay_for_nothing

    CJA_NOOP: Final[int] = 0
    CJA_MOVE_LEFT: Final[int] = 1
    CJA_MOVE_RIGHT: Final[int] = 2
    CJA_MOVE_UP: Final[int] = 3
    CJA_MOVE_DOWN: Final[int] = 4
    CJA_MOVE_LEFT_UP: Final[int] = 5
    CJA_MOVE_RIGHT_UP: Final[int] = 6
    CJA_MOVE_LEFT_DOWN: Final[int] = 7
    CJA_MOVE_RIGHT_DOWN: Final[int]= 8
    CJA_NUM_EXPLICIT_ACTIONS = 9
    """
    if V1 or V2:
        if action in [0, 6]:
            return 3
        elif action in [1, 3, 7]:
            return 1
        elif action in [2, 4, 8]:
            return 2
        elif action in [5, 9]:
            return 4
    elif KD:
        if action in [0, 2]:
            return 1
        elif action in [1, 3]:
            return 2
    elif Dodge:
        if action in [0]:
            return 3
        elif action in [1]:
            return 0


def extract_for_explaining(coin_jump):
    """
    extract state to metric
    input: coin_jump instance
    output: extracted_state to be explained

    x:  agent, key, door, enemy, position_X, position_Y
    y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)

    To be changed when using object-detection tech
    """
    # TODO
    num_of_feature = 6
    num_of_object = 4
    representation = coin_jump.level.get_representation()
    extracted_states = np.zeros((num_of_object, num_of_feature))
    for entity in representation["entities"]:
        if entity[0].name == 'PLAYER':
            extracted_states[0][0] = 1
            extracted_states[0][-2:] = entity[1:3]
            # 27 is the width of map, this is normalization
            # extracted_states[0][-2:] /= 27
        elif entity[0].name == 'KEY':
            extracted_states[1][1] = 1
            extracted_states[1][-2:] = entity[1:3]
            # extracted_states[1][-2:] /= 27
        elif entity[0].name == 'DOOR':
            extracted_states[2][2] = 1
            extracted_states[2][-2:] = entity[1:3]
            # extracted_states[2][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY':
            extracted_states[3][3] = 1
            extracted_states[3][-2:] = entity[1:3]
            # extracted_states[3][-2:] /= 27

    # num_of_feature = 6
    # num_of_object = 5
    # representation = coin_jump.level.get_representation()
    # extracted_states = np.zeros((num_of_object, num_of_feature))
    # for entity in representation["entities"]:
    #     if entity[0].name == 'PLAYER':
    #         extracted_states[0][0] = 1
    #         extracted_states[0][-2:] = entity[1:3]
    #         # 27 is the width of map, this is normalization
    #         # extracted_states[0][-2:] /= 27
    #     elif entity[0].name == 'KEY':
    #         extracted_states[1][1] = 1
    #         extracted_states[1][-2:] = entity[1:3]
    #         # extracted_states[1][-2:] /= 27
    #     elif entity[0].name == 'DOOR':
    #         extracted_states[2][2] = 1
    #         extracted_states[2][-2:] = entity[1:3]
    #         # extracted_states[2][-2:] /= 27
    #     elif entity[0].name == 'GROUND_ENEMY':
    #         extracted_states[3][3] = 1
    #         extracted_states[3][-2:] = entity[1:3]
    #         # extracted_states[3][-2:] /= 27
    #     elif entity[0].name == 'KEY2':
    #         extracted_states[4][1] = 1
    #         extracted_states[4][-2:] = entity[1:3]
    #         # extracted_states[3][-2:] /= 27
    if sum(extracted_states[:, 1]) == 0:
        key_picked = True
    else:
        key_picked = False

    def simulate_prob(extracted_states, num_of_objs, key_picked):
        for i, obj in enumerate(extracted_states):
            obj = add_noise(obj, i, num_of_objs)
            extracted_states[i] = obj
        if key_picked:
            extracted_states[:, 1] = 0
        return extracted_states

    def add_noise(obj, index_obj, num_of_objs):
        mean = torch.tensor(0.2)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)
    states = torch.tensor(np.array(extracted_states), dtype=torch.float32, device="cuda:0").unsqueeze(0)
    return states


def show_explaining(prediction, KD=False, Dodge=False, V1=False, V2=False):
    if KD:
        prednames = ['left_go_get_key', 'right_go_get_key', 'left_go_to_door',
                     'right_go_to_door']
    elif Dodge:
        prednames = ['jump', 'stay']
    elif V1:
        prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
                     'right_go_to_door', 'stay']
    elif V2:
        prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
                     'right_go_to_door', 'stay', 'jump_over_door', 'left_for_nothing', 'right_go_to_enemy',
                     'stay_for_nothing']
    pred = prednames[torch.argmax(prediction).cpu().item()]
    return pred


def plot_weight(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(15, 5))
    plt.ylim([0, 1])
    # x_label = ['jump', 'left_key', 'right_key', 'left_door',
    #            'right_door', 'stay']
    x_label = ['jump', 'left_key', 'right_key', 'left_door',
               'right_door', 'stay', 'jump_door', 'left_nothing', 'right_enemy',
               'stay_nothing']

    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()
        plt.bar(range(1, len(W_) + 1), W_, alpha=1, label='C' + str(i))

    plt.xticks(range(1, len(x_label) + 1), x_label)
    plt.ylabel('Weights')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_weights(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(15, 5))
    plt.ylim([0, 1])
    x_label = ['Jump', 'Left_key', 'Right_key', 'Left_door',
               'Right_door', 'Stay', 'Jump_door', 'Left_nothing', 'Right_enemy',
               'Stay_nothing']
    # x_label = ['Jump', 'Left_k', 'Right_k', 'Left_d',
    #            'Right_d', 'Stay', 'Jump_d', 'Left_n', 'Right_e',
    #            'Stay_n']
    x = np.arange(len(x_label))
    width = 0.15
    X = x - width * 3

    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()

        X = X + width
        plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
        # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))

    plt.xticks(x, x_label, fontproperties="Microsoft YaHei", size=12)
    plt.ylabel('Weights', size=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()

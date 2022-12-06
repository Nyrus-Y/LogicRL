import torch
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

from .facts_converter import FactsConverter
from .nsfr import NSFReasoner
from .logic_utils import build_infer_module, get_lang
from .valuation import RLValuationModule
from .valuation_bf import BFValuationModule

device = torch.device('cpu')


def get_nsfr_model(args, train=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')
    # TODO
    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.m, args.rules)
    if args.m == 'coinjump':
        VM = RLValuationModule(lang=lang, device=device)
    elif args.m == 'bigfish':
        VM = BFValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    m = len(clauses)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return NSFR


def get_nsfr(mode, rule):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, mode, rule)
    if mode == 'coinjump':
        VM = RLValuationModule(lang=lang, device=device)
    elif mode == 'bigfish':
        VM = BFValuationModule(lang=lang, device=device)
    FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    m = len(clauses)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=True, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=True)
    return NSFR


def explaining_nsfr(NSFR, extracted_states):
    V_T = NSFR(extracted_states)
    # prednames = NSFR.prednames
    predicts = NSFR.predict_multi(v=V_T)
    explaining = NSFR.print_explaining(predicts)
    return explaining


def get_predictions(extracted_states, NSFR):
    V_T = NSFR(extracted_states)
    predicts = NSFR.print_explaining(V_T)
    return predicts


def extract_for_cgen_explaining(coin_jump):
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
        mean = torch.tensor(0.1)
        std = torch.tensor(0.05)
        noise = torch.abs(torch.normal(mean=mean, std=std)).item()
        rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
        rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
        rand_noises.insert(index_obj, 1 - noise)

        for i, noise in enumerate(rand_noises):
            obj[i] = rand_noises[i]
        return obj

    extracted_states = simulate_prob(extracted_states, num_of_object, key_picked)

    return torch.tensor(extracted_states, device="cuda:0")


def show_explaining(prediction, prednames, KD=False, Dodge=False, V1=False, V2=False):
    # if KD:
    #     prednames = ['left_go_get_key', 'right_go_get_key', 'left_go_to_door',
    #                  'right_go_to_door']
    # elif Dodge:
    #     prednames = ['jump', 'stay']
    # elif V1:
    #     prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
    #                  'right_go_to_door', 'stay']
    # elif V2:
    #     prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
    #                  'right_go_to_door', 'stay', 'jump_over_door', 'left_for_nothing', 'right_go_to_enemy',
    #                  'stay_for_nothing']
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
    x_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15'
        , 'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10', 'LK11', 'LK12', 'LK13', 'LK14', 'LK15'
        , 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10', 'LD11', 'LD12', 'LD13', 'LD14', 'LD15'
        , 'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'Rk7', 'RK8', 'Rk9', 'RK10', 'RK11', 'RK12', 'Rk13', 'Rk14', 'Rk15'
        , 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10', 'RD11', 'RD12', 'RD13', 'RD14', 'RD15']
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


def plot_weights_multi(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(5, 12))
    plt.xlim([0, 1])
    # x_label = ['Jump', 'Left_key', 'Right_key', 'Left_door',
    #            'Right_door', 'Stay', 'Jump_door', 'Left_nothing', 'Right_enemy',
    #            'Stay_nothing']
    # y_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15'
    #     , 'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10', 'LK11', 'LK12', 'LK13', 'LK14', 'LK15'
    #     , 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10', 'LD11', 'LD12', 'LD13', 'LD14', 'LD15'
    #     , 'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'RK7', 'RK8', 'RK9', 'RK10', 'RK11', 'RK12', 'RK13', 'RK14', 'RK15'
    #     , 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10', 'RD11', 'RD12', 'RD13', 'RD14', 'RD15']
    # y_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10'
    #     , 'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10'
    #     , 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10'
    #     , 'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'RK7', 'RK8', 'RK9', 'RK10'
    #     , 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10']
    # y_label = ['J1', 'J2', 'J3'
    #     , 'LK1', 'LK2', 'LK3'
    #     , 'LD1', 'LD2', 'LD3'
    #     , 'RK1', 'RK2', 'RK3'
    #     , 'RD1', 'RD2', 'RD3']
    y_label = ['Jump', 'Left_k', 'Left_d',
               'Right_k', 'Right_d']
    y = np.arange(len(y_label))
    width = 0.5
    # X = x - width * 3
    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()

        # X = X + width
        # plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
        plt.barh(y=y, width=W_, alpha=1, label='C' + str(i))
        # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))

    plt.yticks(y, y_label, fontproperties="Microsoft YaHei", size=8)
    plt.xlabel('Weights', size=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

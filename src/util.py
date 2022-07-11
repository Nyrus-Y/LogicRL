import torch
import numpy as np
import sys

from src.logic_utils import get_lang
from src.nsfr_utils import get_nsfr_model


def explaining_nsfr(extracted_states, lark_path, lang_base_path):
    device = torch.device('cpu')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'coinjump', 'coinjump')
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)
    V_T = NSFR(extracted_states)

    return NSFR.print_explaining()


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
            print(extracted_states[0][0])
            extracted_states[0][0] = 1
            extracted_states[0][-2:] = entity[1:3]
            # 27 is the width of map, this is normalization
            extracted_states[0][-2:] /= 27
        elif entity[0].name == 'KEY':
            extracted_states[1][1] = 1
            extracted_states[1][-2:] = entity[1:3]
            extracted_states[1][-2:] /= 27
        elif entity[0].name == 'DOOR':
            extracted_states[2][2] = 1
            extracted_states[2][-2:] = entity[1:3]
            extracted_states[2][-2:] /= 27
        elif entity[0].name == 'GROUND_ENEMY':
            extracted_states[3][3] = 1
            extracted_states[3][-2:] = entity[1:3]
            extracted_states[3][-2:] /= 27
    states = torch.tensor(np.array(extracted_states), dtype=torch.float32).unsqueeze(0)
    return states

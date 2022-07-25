import torch
import numpy as np
import torch.nn as nn

from src.logic_utils import get_lang

from src.percept import SlotAttentionPerceptionModule, YOLOPerceptionModule
from src.facts_converter import FactsConverter
from src.nsfr import NSFReasoner
from src.logic_utils import build_infer_module, generate_atoms
from src.valuation import RLValuationModule, RLValuationModule_D, RLValuationModule_KD
from src.coinjump.coinjump.actions import coin_jump_actions_from_unified, CoinJumpActions

device = torch.device('cpu')


def get_nsfr_model(lang, clauses, atoms, bk, device):
    # PM = YOLOPerceptionModule(e=4, d=11, device=device)
    PM = None
    VM = RLValuationModule(
        lang=lang, device=device)
    FC = FactsConverter(lang=lang, perception_module=PM,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                       infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)
    return NSFR


def get_2_nsfr_model(lang, clauses, atoms, bk, device):
    PM = YOLOPerceptionModule(e=4, d=11, device=device)
    VM_D = RLValuationModule_D(
        lang=lang, device=device)
    VM_KD = RLValuationModule_KD(
        lang=lang, device=device
    )

    def combine(VM_D, VM_KD):
        # TODO
        for module in VM_KD.layers:
            if module not in VM_D.layers:
                print(module == VM_D.layers[0])
                VM_D.layers.append(module)

        return VM_D

    VM = combine(VM_D, VM_KD)
    FC = FactsConverter(lang=lang, perception_module=PM,
                        valuation_module=VM, device=device)
    IM = build_infer_module(clauses, atoms, lang,
                            m=len(clauses), infer_step=2, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(perception_module=PM, facts_converter=FC,
                       infer_module=IM, atoms=atoms, bk=bk, clauses=clauses)

    return NSFR


def explaining_nsfr(extracted_states, env, prednames):
    lark_path = '../src/lark/exp.lark'
    lang_base_path = '../data/lang/'

    device = torch.device('cpu')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, env, 'coinjump')
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)

    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(
        v=V_T, prednames=prednames)

    # predicts = NSFR.predict_multi(
    #     v=V_T, prednames=['jump', 'left', 'right'])

    explaining = NSFR.print_explaining(predicts)

    return explaining


def get_predictions(extracted_states, env, prednames):
    lark_path = 'E:\\Github\\Use Knowledge Representation and Reasoning for the policy\\src\\lark\\exp.lark'
    lang_base_path = 'E:\\Github\\Use Knowledge Representation and Reasoning for the policy\\data\\lang\\'

    device = torch.device('cpu')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, env, 'coinjump')
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)

    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(
        v=V_T, prednames=prednames)

    return predicts


def explaining_nsfr_combine(extracted_states, env1, env2):
    lark_path = '../src/lark/exp.lark'
    lang_base_path = '../data/lang/'

    device = torch.device('cpu')

    def combine(data1, data2):
        lang = data1['lang']
        for pred in data2['lang'].preds:
            if pred not in data1['lang'].preds:
                lang.preds.append(pred)
        clauses = data1['clauses']
        for clause in data2['clauses']:
            if clause not in data1['clauses']:
                clauses.append(clause)
        bks = data1['bk']
        for bk in data2['bk']:
            if bk not in data1['bk']:
                bks.append(bk)
        atoms = data1['atoms']
        for atom in data2['atoms']:
            if atom not in data1['bk']:
                atoms.append(atom)
        return lang, clauses, bks, atoms

    lang1, clauses1, bk1, atoms1 = get_lang(
        lark_path, lang_base_path, env1, 'coinjump')
    lang2, clauses2, bk2, atoms2 = get_lang(
        lark_path, lang_base_path, env2, 'coinjump')
    data1 = {'lang': lang1, 'clauses': clauses1, 'bk': bk1, 'atoms': atoms1}
    data2 = {'lang': lang2, 'clauses': clauses2, 'bk': bk2, 'atoms': atoms2}
    lang, clauses, bk, atoms = combine(data1, data2)
    NSFR = get_2_nsfr_model(lang, clauses, atoms, bk, device)

    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(
        v=V_T, prednames=['left_go_get_key', 'right_go_get_key', 'left_go_to_door',
                          'right_go_to_door'])

    # predicts = NSFR.predict_multi(
    #     v=V_T, prednames=['jump', 'left', 'right'])

    explaining = NSFR.print_explaining(predicts)

    return explaining


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


def num_action_select(action):
    """
    0:jump
    1:left_go_get_key
    2:right_go_get_key
    3:left_go_to_door
    4:right_go_to_door
    5:stay
    """
    if action == 0:
        return 3
    elif action == 1 or 3:
        return 1
    elif action == 2 or 4:
        return 2
    elif action == 5:
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

    states = torch.tensor(np.array(extracted_states), dtype=torch.float32).unsqueeze(0)
    return states


def reward_shaping(reward, last_coinjump, action):
    return reward

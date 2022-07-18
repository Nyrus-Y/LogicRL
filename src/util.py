import torch
import numpy as np
import sys

from src.logic_utils import get_lang

from src.percept import SlotAttentionPerceptionModule, YOLOPerceptionModule
from src.facts_converter import FactsConverter
from src.nsfr import NSFReasoner
from src.logic_utils import build_infer_module, generate_atoms
from src.valuation import RLValuationModule
from src.coinjump.coinjump.actions import coin_jump_actions_from_unified, CoinJumpActions

device = torch.device('cpu')


def get_nsfr_model(lang, clauses, atoms, bk, device):
    PM = YOLOPerceptionModule(e=4, d=11, device=device)
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


def explaining_nsfr(extracted_states):
    lark_path = '../src/lark/exp.lark'
    lang_base_path = '../data/lang/'

    device = torch.device('cpu')
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'coinjump', 'coinjump')
    NSFR = get_nsfr_model(lang, clauses, atoms, bk, device)

    V_T = NSFR(extracted_states)
    predicts = NSFR.predict_multi(
        v=V_T, prednames=['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door', 'right_go_to_door'])

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

    return action


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

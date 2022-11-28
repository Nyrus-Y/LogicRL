import torch
import numpy as np

from .coinjump.coinjump.actions import coin_jump_actions_from_unified, CoinJumpActions


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


def num_action_select(action, prednames):
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
    if prednames[action] in ['jump']:
        return 3
    elif prednames[action] in ['left_go_get_key', 'left_go_to_door']:
        return 1
    elif prednames[action] in ['right_go_get_key', 'right_go_to_door']:
        return 2


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


def num_action_select_bm(action, V1=False, V2=False):
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
        elif action in [1, 2, 7]:
            return 1
        elif action in [3, 4, 8]:
            return 2


def prediction_to_action_cj(prediction, args, prednames=None):
    if args.alg == 'ppo':
        # simplified action--- only left right up
        # action = coin_jump_actions_from_unified(torch.argmax(predictions).cpu().item() + 1)
        action = prediction + 1
    elif args.alg == 'logic':
        if not isinstance(prediction, int):
            prediction = torch.argmax(prediction).cpu().item()
        action = num_action_select(prediction, prednames)
    return action

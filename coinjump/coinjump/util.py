import torch
import numpy as np


def extract_for_explaining(coin_jump):
    """
    extract state to metric
    input: coin_jump instance
    output: extracted_state to be explained

    x:  agent, key, door, enemy1, position_X, position_Y
    y:  obj1(agent), obj2(key), obj3(door)，obj4(enemy)

    To be changed when using object-detection tech
    """
    # TODO
    num_of_feature = 6
    repr = coin_jump.level.get_representation()
    extracted_states = []
    for i, entity in enumerate(repr["entities"]):
        extracted_state = [0] * num_of_feature
        extracted_state[i] = 1
        extracted_state[-2:] = entity[1:3]
        extracted_states.append(extracted_state)

    return torch.tensor(np.array(extracted_states))

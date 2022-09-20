import numpy
import numpy as np
import torch
import math


def extract_state(obs):
    state = obs.reshape(-1)
    # return torch.tensor(state, device='cuda:0')
    return state.tolist()


def fuzzy_position(pos1, pos2, keyword):
    x = pos2[:, 0] - pos1[:, 0]
    y = pos2[:, 1] - pos1[:, 1]
    tan = torch.atan2(y, x)
    degree = tan[:] / torch.pi * 180

    if keyword == 'top':
        probs = 1 - abs(degree[:] - 90) / 45
        result = torch.where((135 >= degree) & (degree >= 45), probs, 0)
    elif keyword == 'top_left':
        probs = 1 - abs(degree[:] - 135) / 45
        result = torch.where((180 >= degree) & (degree >= 90), probs, 0)
    elif keyword == 'left':
        probs = 1 - abs(degree[:] - 180) / 45
        result = torch.where((degree <= -135) & (degree >= 135), probs, 0)
    elif keyword == 'bottom_left':
        probs = 1 - abs(degree[:] + 135) / 45
        result = torch.where((-90 >= degree) & (degree >= -180), probs, 0)
    elif keyword == 'bottom':
        probs = 1 - abs(degree[:] + 90) / 45
        result = torch.where((-45 >= degree) & (degree >= -135), probs, 0)
    elif keyword == 'bottom_right':
        probs = 1 - abs(degree[:] + 45) / 45
        result = torch.where((0 >= degree) & (degree >= -90), probs, 0)
    elif keyword == 'right':
        probs = 1 - abs(degree[:]) / 45
        result = torch.where((45 >= degree) & (degree >= -45), probs, 0)
    elif keyword == 'top_right':
        probs = 1 - abs(degree[:] - 45) / 45
        result = torch.where((90 >= degree) & (degree >= 0), probs, 0)
    return result

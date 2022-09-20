import numpy
import numpy as np
import torch

def extract_state(obs):
    state = obs.reshape(-1)
    # return torch.tensor(state, device='cuda:0')
    return state.tolist()

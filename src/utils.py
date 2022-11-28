import random
import numpy as np
import torch


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    # environment.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Set all environment deterministic to seed {seed}")


def env_loader(args):
    return
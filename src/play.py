import argparse
import numpy as np
import torch
from environments.bigfish
from config import *
import random


def make_deterministic(seed, environment):
    np.random.seed(seed)
    environment.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all environment deterministic to seed {seed}")
    random.seed(random_seed)


parser = argparse.ArgumentParser()

parser.add_argument("-alg", "--algo", help="Activation function to use",
                    action="store", dest="algo", required=True,
                    choices=['PPO', 'SimplePPO', 'BeamRules'])
parser.add_argument("-env", "--environment", help="Environment to train on",
                    required=True, action="store", dest="env",
                    choices=['coinjump', 'bigfish', 'heist'])
parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                    required=True, action="store", dest="seed", type=int,
                    default=0)
parser.add_argument("--recover", help="Recover from the last trained agent",
                    action="store_true", dest="recover", default=False)
parser.add_argument("--load", help="Pytorch file to load if continue training",
                    action="store_true", dest="load", default=False)


args = parser.parse_args()


#### Continue to render

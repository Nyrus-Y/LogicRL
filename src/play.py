import argparse
import numpy as np
import torch
import random
import os
import pathlib

from utils import make_deterministic
from utils_game import render_coinjump, render_bigfish, render_heist
from agents.neural_agent import ActorCritic
from agents.logic_agent import NSFR_ActorCritic


def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        if args.alg == 'ppo':
            model = ActorCritic(args)
        elif args.alg == 'logic':
            model = NSFR_ActorCritic(args)
        model.load_state_dict(state_dict=torch.load(f))

    model = model.actor
    model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help="Seed for pytorch + env", default=0,
                        required=False, action="store", dest="seed", type=int)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['coinjump', 'bigfish', 'heist'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['CoinJumpEnvLogic-v0', 'CoinJumpEnvNeural-v0',
                                 'bigfishm', 'bigfishc', 'heist'])
    parser.add_argument("-r", "--rules", dest="rules", default=None,
                        required=False, choices=['coinjump_5a', 'bigfish_simplified_actions', 'heist'])
    parser.add_argument("-mo", "--model_file", dest="model_file", default=None)

    # parser.add_argument("--recover", help="Recover from the last trained agent",
    #                     action="store_true", dest="recover", default=False)
    # parser.add_argument("--load", help="Pytorch file to load if continue training",
    #                     action="store_true", dest="load", default=False)
    # args = ['-s 1', '-alg ppo', '-m coinjump', '-env CoinJumpEnvNeural']
    # args = ['-m', 'bigfish', '-alg', 'logic', '-env', 'bigfishm','-r','bigfish_simplified_actions']
    # args = parser.parse_args(args)
    args = parser.parse_args()

    # fix seed
    # seed = random.randint(0, 123456)
    make_deterministic(args.seed)

    # load trained_model
    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        # model_name = input('Enter file name: ')
        #
        # model_file = os.path.join(current_path, 'models', args.m, args.alg, model_name)
        model_file = "/home/quentin/Documents/logicRL/src/models/coinjump/ppo/ppo_seed_0_epi_34390.pth"

    else:
        model_file = pathlib.Path(args.model_file)

    model = load_model(model_file, args)

    #### Continue to render
    if args.m == 'coinjump':
        render_coinjump(model, args)
    elif args.m == 'bigfish':
        render_bigfish(model, args)
    elif args.m == 'heist':
        render_heist(model, args)


if __name__ == "__main__":
    main()

import argparse
import torch
import os

from utils import make_deterministic
from utils_game import render_coinjump, render_bigfish, render_heist
from agents.neural_agent import ActorCritic, NeuralPlayer
from agents.logic_agent import NSFR_ActorCritic, LogicPlayer
from agents.random_agent import RandomPlayer

device = torch.device('cuda:0')


def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        if args.alg == 'ppo':
            model = ActorCritic(args).to(device)
        elif args.alg == 'logic':
            model = NSFR_ActorCritic(args).to(device)
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
                        choices=['ppo', 'logic', 'random'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['coinjump', 'bigfish', 'heist'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['CoinJumpEnv-v1', 'bigfishm', 'bigfishc', 'eheist'])
    parser.add_argument("-r", "--rules", dest="rules", default=None, required=False,
                        choices=['coinjump_5a', 'coinjump_bs', 'bigfish_simplified_actions', 'bigfishc', 'eheist_1',
                                 'eheist_2'])
    parser.add_argument("-mo", "--model_file", dest="model_file", default=None)
    # arg = ['-m', 'heist', '-alg', 'logic', '-env', 'eheist', '-r', '']
    args = parser.parse_args()
    # fix seed
    make_deterministic(args.seed)

    # load trained_model
    if args.alg != 'random':
        # read filename from models
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'models', args.m, args.alg, model_name)
        model = load_model(model_file, args)
    else:
        model = None

    #### create agent
    if args.alg == 'ppo':
        agent = NeuralPlayer(args, model)
    elif args.alg == 'logic':
        agent = LogicPlayer(args, model)
    elif args.alg == 'random':
        agent = RandomPlayer(args)

    #### Continue to render
    if args.m == 'coinjump':
        render_coinjump(agent, args)
    elif args.m == 'bigfish':
        render_bigfish(agent, args)
    elif args.m == 'heist':
        render_heist(agent, args)


if __name__ == "__main__":
    main()

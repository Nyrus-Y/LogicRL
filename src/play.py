import argparse
import torch
import os

from utils import make_deterministic
from utils_game import render_coinjump, render_bigfish, render_heist, render_ecoinrun
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
                        choices=['ppo', 'logic', 'random', 'human'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['coinjump', 'bigfish', 'heist', 'ecoinrun'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['CoinJumpEnv-v1', 'CoinJumpEnv-v2',
                                 'bigfishm', 'bigfishc',
                                 'eheist', 'eheistc1', 'eheistc2',
                                 'ecoinrun'])
    parser.add_argument("-r", "--rules", dest="rules", default=None,
                        required=False,
                        choices=['coinjump_human_assisted', 'coinjump_10a', 'coinjump_bs_top10', 'coinjump_bs_top1',
                                 'coinjump_bs_top3',
                                 'bigfish_human_assisted', 'bigfishc', 'bigfishm_bs_top5', 'bigfishm_bs_top3',
                                 'bigfishm_bs_top1', 'more_redundant_actions',
                                 'eheist_human_assisted', 'eheist_bs_top5', 'eheist_bs_top1',
                                 ])
    parser.add_argument("-l", "--log", help="record the information of games", type=bool, default=False, dest="log")
    parser.add_argument("--log_file_name", help="the name of log file", required=False, dest='logfile')
    parser.add_argument("--render", help="render the game",type=bool, default=True, dest="render")
    args = parser.parse_args()

    # fix seed
    make_deterministic(args.seed)

    # load trained_model
    if args.alg not in ['random', 'human']:
        # read filename from models
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'models', args.m, args.alg, model_name)
        model = load_model(model_file, args)
    else:
        model = None

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    if args.log:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + args.m + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        #### get number of log files in log directory
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)

        #### create new log file for each run
        log_f_name = log_dir + args.alg + '_' + args.env + "_log_" + str(run_num) + ".csv"
        args.logfile = log_f_name
        print("current logging run number for " + args.env + " : ", run_num)
        print("logging at : " + log_f_name)

    #### create agent
    if args.alg == 'ppo':
        agent = NeuralPlayer(args, model)
    elif args.alg == 'logic':
        agent = LogicPlayer(args, model)
    elif args.alg == 'random':
        agent = RandomPlayer(args)
    elif args.alg == 'human':
        agent = 'human'

    #### Continue to render
    if args.m == 'coinjump':
        render_coinjump(agent, args)
    elif args.m == 'bigfish':
        render_bigfish(agent, args)
    elif args.m == 'heist':
        render_heist(agent, args)
    elif args.m == 'ecoinrun':
        render_ecoinrun(agent, args)


if __name__ == "__main__":
    main()

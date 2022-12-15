import argparse
import torch
import os
import json
from nsfr.utils_beam import get_nsfr_model
from nsfr.logic_utils import get_lang
from nsfr.mode_declaration import get_mode_declarations
from nsfr.clause_generator import ClauseGenerator
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.terminated = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.terminated[:]
        del self.predictions[:]

    def load_buffer(self, args):
        current_path = os.path.dirname(__file__)
        path = os.path.join(current_path, 'data', args.d)
        with open(path, 'r') as f:
            state_info = json.load(f)

        self.actions = torch.tensor(state_info['actions']).to(device)
        self.logic_states = torch.tensor(state_info['logic_states']).to(device)
        self.neural_states = torch.tensor(state_info['neural_states']).to(device)
        self.action_probs = torch.tensor(state_info['action_probs']).to(device)
        self.logprobs = torch.tensor(state_info['logprobs']).to(device)
        self.rewards = torch.tensor(state_info['reward']).to(device)
        self.terminated = torch.tensor(state_info['terminated']).to(device)
        self.predictions = torch.tensor(state_info['predictions']).to(device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument('-r', "--rules", required=True, help="choose to root rules", dest='r',
                        choices=["coinjump_root", 'bigfishm_root', 'eheist_root'])
    parser.add_argument('-m', "--model", required=True, help="the game mode for beam-search", dest='m',
                        choices=['coinjump', 'bigfish', 'heist'])
    parser.add_argument('-t', "--t-beam", type=int, default=3, help="Number of rule expantion of clause generation.")
    parser.add_argument('-n', "--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--s", type=int, default=1, help="The size of the logic program.")
    parser.add_argument('--scoring', type=bool, help='beam search rules with scored rule by trained ppo agent',
                        default=False, dest='scoring')
    parser.add_argument('-d', '--dataset', required=False, help='the dataset to load if scoring', dest='d',
                        choices=['coinjump.json', 'heist.json'])
    arg = ['-m', 'heist', '-r', 'eheist_root', '--scoring', 'True', '-d', 'heist.json','-n','12']
    # arg = ['-m', 'coinjump', '-r', 'coinjump_root', '--scoring', 'True', '-d', 'coinjump.json']
    args = parser.parse_args(arg)
    return args


def run():
    args = get_args()
    # load state info for searching if scoring
    if args.scoring:
        buffer = RolloutBuffer()
        buffer.load_buffer(args)
    # writer = SummaryWriter(f"runs/{env_name}", purge_step=0)
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, '../nsfr/nsfr', 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, '../nsfr/nsfr', 'data/lang/')

    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, args.m, args.r)
    bk_clauses = []
    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    mode_declarations = get_mode_declarations(args, lang)

    print('get mode_declarations')
    if args.scoring:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, buffer=buffer, device=device)
    else:
        cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, mode_declarations, device=device)

    clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")


if __name__ == "__main__":
    run()

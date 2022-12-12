import argparse
import torch
import os

from nsfr.nsfr_utils import get_nsfr_model
from nsfr.logic_utils import get_lang
from nsfr.mode_declaration import get_mode_declarations
from nsfr.clause_generator import ClauseGenerator
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:0')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["coinjump_5a"],
                        help="Use coinjump dataset")
    parser.add_argument("--dataset-type", default="coinjump", help="coinjump env")
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=3, help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    args = parser.parse_args()
    return args


def run():
    args = get_args()
    env_name = 'coinjump_search'
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    writer = SummaryWriter(f"runs/{env_name}", purge_step=0)

    # lang, clauses, bk, atoms = get_lang(
    #     lark_path, lang_base_path, args.dataset_type, args.dataset)
    lang, clauses, bk, atoms = get_lang(
        lark_path, lang_base_path, 'coinjump', env_name)
    # clauses = update_initial_clauses(clauses, args.n_obj)
    bk_clauses = []
    # print("clauses: ", clauses)

    # Neuro-Symbolic Forward Reasoner for clause generation
    NSFR_cgen = get_nsfr_model(args, lang, clauses, atoms, bk, bk_clauses, device=device)  # torch.device('cpu'))
    mode_declarations = get_mode_declarations(args, lang, args.n_obj)
    # print("Maze terminated")

    cgen = ClauseGenerator(args, NSFR_cgen, lang, atoms, buffer, mode_declarations, device=device,
                           no_xil=args.no_xil)  # torch.device('cpu'))
    # generate clauses
    # if args.pre_searched:
    #     clauses = get_searched_clauses(lark_path, lang_base_path, args.dataset_type, args.dataset)
    # else:
    clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
    print("====== ", len(clauses), " clauses are generated!! ======")


if __name__ == "__main__":
    run()

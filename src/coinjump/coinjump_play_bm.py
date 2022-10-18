import random
import time
import argparse
import numpy as np
import torch
import os
from src.coinjump.coinjump.imageviewer import ImageViewer
from src.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from src.coinjump.coinjump.coinjump.coinjump import CoinJump
from src.util import extract_for_explaining, explaining_nsfr, action_select, get_nsfr

from src.nsfr_utils import get_nsfr_model, update_initial_clauses
from src.logic_utils import get_lang, get_searched_clauses
from src.mode_declaration import get_mode_declarations
from src.clause_generator import ClauseGenerator

KEY_r = 114
device = torch.device('cuda:0')


def setup_image_viewer(coinjump):
    viewer = ImageViewer(
        "coinjump1",
        coinjump.camera.height,
        coinjump.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_coinjump_instance():
    seed = random.random()
    coin_jump = CoinJump(start_on_first_action=True, V1=True)
    # level_generator = DummyGenerator()
    # change generator to choose env
    level_generator = ParameterizedLevelGenerator_V1()
    level_generator.generate(coin_jump, seed=seed)
    # level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["coinjump", "coinjump_5a", "coinjumpD", "coinjumpKD", ],
                        help="Use coinjump dataset")
    parser.add_argument("--dataset-type", default="coinjump", help="coinjump env")
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--small-data", action="store_true", help="Use small training data.")
    parser.add_argument("--no-xil", action="store_true", help="Do not use confounding labels for clevr-hans.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=4, help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=5, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200, help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true", help="Using pre searched clauses.")
    args = parser.parse_args()
    return args


def run():
    coin_jump = create_coinjump_instance()
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0
    last_explaining = None

    args = get_args()

    env_name = 'coinjump_5a'
    nsfr = get_nsfr(env_name)

    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    device = torch.device('cuda:0')
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

    while True:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration

        # step game
        action = []

        if KEY_r in viewer.pressed_keys:
            coin_jump = create_coinjump_instance()
            print("--------------------------     next game    --------------------------")

        if not coin_jump.level.terminated:

            # extract state for explaining
            extracted_state = extract_for_explaining(coin_jump)
            # explaining = explaining_nsfr(nsfr, extracted_state)
            # action = action_select(explaining)

            cgen = ClauseGenerator(args, NSFR_cgen, lang, extracted_state, mode_declarations, bk_clauses, device=device,
                                   no_xil=args.no_xil)  # torch.device('cpu'))
            # generate clauses
            # if args.pre_searched:
            #     clauses = get_searched_clauses(lark_path, lang_base_path, args.dataset_type, args.dataset)
            # else:
            clauses = cgen.generate(clauses, T_beam=args.t_beam, N_beam=args.n_beam, N_max=args.n_max)
            print("====== ", len(clauses), " clauses are generated!! ======")

            if last_explaining is None:
                print(explaining)
                last_explaining = explaining
            elif explaining != last_explaining:
                print(explaining)
                last_explaining = explaining

        reward = coin_jump.step(action)
        score = coin_jump.get_score()
        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        terminated = coin_jump.level.terminated
        # if terminated:
        #     print("score = ", score)
        if viewer.is_escape_pressed:
            break

    print("Maze terminated")


if __name__ == "__main__":
    run()

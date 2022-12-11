import random
import tqdm
import time
import argparse
import numpy as np
import torch
import os
from argparse import ArgumentParser
import pathlib

from environments.coinjump.coinjump.imageviewer import ImageViewer
from environments.coinjump.coinjump.coinjump.coinjump import CoinJump
from environments.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from agents.utils_coinjump import extract_state, sample_to_model_input, collate
from agents.neural_agent import ActorCritic
from environments.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified

from nsfr.utils import extract_for_cgen_explaining
from nsfr.nsfr_utils import get_nsfr_model
from nsfr.logic_utils import get_lang
from nsfr.mode_declaration import get_mode_declarations
from nsfr.clause_generator import ClauseGenerator
from torch.utils.tensorboard import SummaryWriter

KEY_r = 114
device = torch.device('cuda:0')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.logic_states = []
        self.neural_states = []
        self.action_probs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.logic_states[:]
        del self.neural_states[:]
        del self.action_probs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size to infer with")
    parser.add_argument("--batch-size-bs", type=int, default=1, help="Batch size in beam search")
    parser.add_argument("--e", type=int, default=6,
                        help="The maximum number of objects in one image")
    parser.add_argument("--dataset", choices=["coinjump_5a"],
                        help="Use coinjump dataset")
    parser.add_argument("--dataset-type", default="coinjump", help="coinjump env")
    parser.add_argument('--device', default='cuda:0',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of threads for data loader")
    parser.add_argument('--gamma', default=0.01, type=float,
                        help='Smooth parameter in the softor function')
    parser.add_argument("--plot", action="store_true",
                        help="Plot images with captions.")
    parser.add_argument("--t-beam", type=int, default=3, help="Number of rule expantion of clause generation.")
    parser.add_argument("--n-beam", type=int, default=8, help="The size of the beam.")
    parser.add_argument("--n-max", type=int, default=50, help="The maximum number of clauses.")
    parser.add_argument("--m", type=int, default=1, help="The size of the logic program.")
    parser.add_argument("--n-obj", type=int, default=2, help="The number of objects to be focused.")
    parser.add_argument("--epochs", type=int, default=101, help="The number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="The learning rate.")
    parser.add_argument("--n-data", type=float, default=200, help="The number of data to be used.")
    parser.add_argument("--pre-searched", action="store_true", help="Using pre searched clauses.")
    args = parser.parse_args()
    return args


def setup_image_viewer(coinjump):
    viewer = ImageViewer(
        "coinjump1",
        coinjump.camera.height,
        coinjump.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_coinjump_instance(seed=None):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    coin_jump = CoinJump(start_on_first_action=True)
    level_generator = ParameterizedLevelGenerator_V1()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play coinjump")
    parser.add_argument("-m", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", type=int)
    args = parser.parse_args()

    # TODO change path of model
    if args.model_file is None:
        # read filename from stdin
        current_path = os.path.dirname(__file__)
        model_name = input('Enter file name: ')
        model_file = os.path.join(current_path, 'models', 'coinjump', 'ppo', model_name)
        # model_file = f"../src/ppo_coinjump_model/{input('Enter file name: ')}"

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, args, set_eval=True):
    with open(model_path, "rb") as f:
        model = ActorCritic(args).to(device)
        model.load_state_dict(state_dict=torch.load(f))
    if isinstance(model, ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def run():
    args, model_file = parse_args()

    model = load_model(model_file, args)

    seed = random.seed() if args.seed is None else int(args.seed)

    coin_jump = create_coinjump_instance(seed=seed)
    # viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    args = get_args()
    buffer = RolloutBuffer()
    # nsfr = get_nsfr(env_name)

    # collect data
    max_states = 10000
    max_states = 100
    save_frequence = 1
    print_frequence = 1000
    step = 0
    collected_states = 0

    while collected_states < max_states:
        # # control framerate
        # current_frame_time = time.time()
        # # limit frame rate
        # if last_frame_time + target_frame_duration > current_frame_time:
        #     sl = (last_frame_time + target_frame_duration) - current_frame_time
        #     time.sleep(sl)
        #     continue
        # last_frame_time = current_frame_time  # save frame start time for next iteration

        # step game
        action = []
        step += 1

        if not coin_jump.level.terminated:
            model_input = sample_to_model_input((extract_state(coin_jump), []))
            model_input = collate([model_input])
            prediction = model(model_input['state'])
            # 1 left 2 right 3 up
            action = coin_jump_actions_from_unified(torch.argmax(prediction).cpu().item() + 1)

            logic_state = extract_for_cgen_explaining(coin_jump)
            if step % save_frequence == 0:
                collected_states += 1
                buffer.logic_states.append(logic_state.detach())
                buffer.actions.append(torch.argmax(prediction.detach()))
                buffer.action_probs.append(prediction.detach())
                buffer.neural_states.append(model_input['state'])

                if collected_states % print_frequence == 0:
                    print("states collected: {} , max states: {}".format(collected_states, max_states))
        else:
            coin_jump = create_coinjump_instance(seed=seed)
            action = []

        reward = coin_jump.step(action)
        # score = coin_jump.get_score()
        # np_img = np.asarray(coin_jump.camera.screen)
        # viewer.show(np_img[:, :, :3])

        # terminated = coin_jump.level.terminated
        # if viewer.is_escape_pressed:
        #     break

    print('data collected')

    env_name = 'coinjump_search'
    current_path = os.getcwd()
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    lang_base_path = os.path.join(current_path, 'data/lang/')

    writer = SummaryWriter(f"runs/{env_name}", purge_step=0)

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

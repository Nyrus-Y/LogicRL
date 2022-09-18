import random
import time
from argparse import ArgumentParser
import pathlib
import pickle
import torch
import torch.nn as nn
import numpy as np
import os

from src.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified
from src.coinjump.coinjump.imageviewer import ImageViewer
from src.util import extract_for_explaining, num_action_select, show_explaining,generate_captions

from src.coinjump.coinjump.coinjump.paramLevelGenerator_keydoor import ParameterizedLevelGenerator_KeyDoor
from src.coinjump.coinjump.coinjump.paramLevelGenerator_dodge import ParameterizedLevelGenerator_Dodge
from src.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from src.coinjump.coinjump.coinjump.paramLevelGenerator_keys import ParameterizedLevelGenerator_Keys
from src.coinjump.coinjump.coinjump.coinjump import CoinJump

from src.coinjump.coinjump_learn.training.ppo_coinjump_logic_policy import NSFR_ActorCritic
from src.coinjump.coinjump_learn.models.mlpCriticController import MLPCriticController

from src.valuation import RLValuationModule
from src.facts_converter import FactsConverter
from src.logic_utils import build_infer_module, get_lang
from src.nsfr_training import NSFReasoner

KEY_r = 114


class NSFR_ActorCritic(nn.Module):
    def __init__(self):
        super(NSFR_ActorCritic, self).__init__()

        self.actor = self.get_nsfr_model()
        self.critic = MLPCriticController(out_size=1)

    def forward(self):
        raise NotImplementedError

    def act(self):
        pass

    def get_nsfr_model(self):
        current_path = os.getcwd()
        lark_path = os.path.join(current_path, 'lark/exp.lark')
        lang_base_path = os.path.join(current_path, 'data/lang/')
        # TODO
        device = torch.device('cuda:0')
        lang, clauses, bk, atoms = get_lang(
            lark_path, lang_base_path, 'coinjump', 'coinjump')

        VM = RLValuationModule(lang=lang, device=device)
        FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
        # m = len(clauses)
        m = 5
        IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=True, device=device)
        # Neuro-Symbolic Forward Reasoner
        NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=True)
        return NSFR


def setup_image_viewer(coinjump):
    viewer = ImageViewer(
        "coinjump1",
        coinjump.camera.height,
        coinjump.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_coinjump_instance(seed=None, V1=False, key_door=False, Dodge=False, keys=False):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    if V1:
        coin_jump = CoinJump(V1=True)
        level_generator = ParameterizedLevelGenerator_V1()
    elif key_door:
        coin_jump = CoinJump(Key_Door_model=True)
        level_generator = ParameterizedLevelGenerator_KeyDoor()
    elif keys:
        coin_jump = CoinJump(keys=True)
        level_generator = ParameterizedLevelGenerator_Keys()
    else:
        coin_jump = CoinJump(Dodge_model=True)
        level_generator = ParameterizedLevelGenerator_Dodge()
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
        model_file = os.path.join(current_path, 'nsfr_coinjump_model', model_name)

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, set_eval=True):
    with open(model_path, "rb") as f:
        model = NSFR_ActorCritic()
        # model = torch.load(f)
        model.load_state_dict(state_dict=torch.load(f))

    if isinstance(model, NSFR_ActorCritic):
        model = model.actor
        model.as_dict = True

    if set_eval:
        model = model.eval()

    return model


def run():
    args, model_file = parse_args()

    model = load_model(model_file)

    seed = random.seed() if args.seed is None else int(args.seed)
    # TODO input parameter to change mode
    coin_jump = create_coinjump_instance(seed=seed, V1=True)
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    last_explaining = None

    while True:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration

        # TODO change for reset env
        if KEY_r in viewer.pressed_keys:
            # coin_jump = create_coinjump_instance(seed=seed, Key_Door_model=True)
            coin_jump = create_coinjump_instance(seed=seed, V1=True)
            print("--------------------------     next game    --------------------------")
            # coin_jump = create_coinjump_instance(seed=seed,Dodge_model=True)
        # step game
        if not coin_jump.level.terminated:

            # extract state for explaining
            extracted_state = extract_for_explaining(coin_jump)
            # explaining = explaining_nsfr(extracted_state)
            #
            # if last_explaining is None:
            #     print(explaining)
            #     last_explaining = explaining
            # elif explaining != last_explaining:
            #     print(explaining)
            #     last_explaining = explaining

            # TODO change sample function of different envs

            prediction = model(extracted_state)
            # prediction[0][0] = 0
            # print(model.state_dict())
            print(show_explaining(prediction, V2=True))
            # print(model.state_dict())
            num = torch.argmax(prediction).cpu().item()
            action = num_action_select(num, V2=True)
            action = coin_jump_actions_from_unified(action)
        else:

            action = []
        reward = coin_jump.step(action)

        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        # terminated = coin_jump.level.terminated
        # if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

    print("Terminated")


def load_recording(replay_file):
    with open(replay_file, 'rb') as f:
        # data = {
        #    'actions': actions, =[ACTION,, ...]
        #    'meta': coinjump1.level.get_representation(),
        #    'score': coinjump1.score
        # }
        data = pickle.load(f)
        print("loading", data)
        return data


if __name__ == "__main__":
    run()

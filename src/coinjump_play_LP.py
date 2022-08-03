import random
import time
from argparse import ArgumentParser
import pathlib
import pickle
import torch
import numpy as np

from src.coinjump.coinjump.actions import coin_jump_actions_from_unified
from src.coinjump.imageviewer import ImageViewer
from src.util import extract_for_explaining, num_action_select, show_explaining

from src.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from src.coinjump.coinjump.coinjump import CoinJump

from src.coinjump_learn.training.data_transform import extract_state, sample_to_model_input_V1, collate
from src.coinjump_learn.training.ppo_coinjump_logic_policy import NSFR_ActorCritic

KEY_r = 114


def setup_image_viewer(coinjump):
    viewer = ImageViewer(
        "coinjump1",
        coinjump.camera.height,
        coinjump.camera.width,
        monitor_keyboard=True,
        # relevant_keys=set('W','A','S','D','SPACE')
    )
    return viewer


def create_coinjump_instance(seed=None, V1=False):
    seed = random.randint(0, 100000000) if seed is None else seed

    # level_generator = DummyGenerator()
    coin_jump = CoinJump(start_on_first_action=True, V1=True)
    level_generator = ParameterizedLevelGenerator_V1()

    level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def parse_args():
    parser = ArgumentParser("Loads a model and lets it play CoinJump")
    parser.add_argument("-m", "--model_file", dest="model_file", default=None)
    parser.add_argument("-s", "--seed", dest="seed", type=int)
    args = parser.parse_args()

    # TODO change path of model
    if args.model_file is None:
        # read filename from stdin
        # model_file = f"../training/PPO_preTrained/CoinJumpEnv-v0/{input('Enter file name: ')}"
        # model_file = f"../training/PPO_preTrained/CoinJumpEnvKD-v0/{input('Enter file name: ')}"
        # model_file = f"../training/PPO_preTrained/CoinJumpEnvDodge-v0/{input('Enter file name: ')}"
        # model_file = f"../training/PPO_preTrained/CoinJumpEnv-v1/{input('Enter file name: ')}"
        model_file = f"../src/nsfr_coinjump_model/{input('Enter file name: ')}"

    else:
        model_file = pathlib.Path(args.model_file)

    return args, model_file


def load_model(model_path, set_eval=True):
    with open(model_path, "rb") as f:
        model = torch.load(f)

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
    # coin_jump = create_coinjump_instance(seed=seed, Key_Door_model=True)
    # coin_jump = create_coinjump_instance(seed=seed, Dodge_model=True)
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
            print(show_explaining(prediction))
            num = torch.argmax(prediction).cpu().item()
            action = num_action_select(num)
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

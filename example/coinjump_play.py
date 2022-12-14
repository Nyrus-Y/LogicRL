# example of logic player
import random
import time
import argparse
import numpy as np
import sys
sys.path.insert(0,'../')
from src.environments.coinjump.coinjump.imageviewer import ImageViewer
from src.environments.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from src.environments.coinjump.coinjump.coinjump.coinjump import CoinJump
from nsfr.utils import get_nsfr_model, get_predictions
from src.agents.utils_coinjump import extract_logic_state_coinjump



KEY_SPACE = 32
KEY_w = 119
KEY_a = 97
KEY_s = 115
KEY_d = 100
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


def create_coinjump_instance():
    seed = random.random()

    coin_jump = CoinJump(start_on_first_action=False)
    # level_generator = DummyGenerator()

    # change generator to choose env
    level_generator = ParameterizedLevelGenerator_V1()

    level_generator.generate(coin_jump, seed=seed)
    # level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def explaining_to_action(explaining):
    if 'jump' in explaining:
        return 3
    elif 'left' in explaining:
        return 1
    elif 'right' in explaining:
        return 2


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=False, action="store", dest="m", default='coinjump',
                        choices=['coinjump'])
    parser.add_argument("-r", "--rules", dest="rules", default='coinjump_5a',
                        required=False, choices=['coinjump_5a'])
    args = parser.parse_args()

    coin_jump = create_coinjump_instance()
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0
    last_explaining = None

    nsfr = get_nsfr_model(args, train=False)
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

            # extract state for expextracted_statelaining
            extracted_state = extract_logic_state_coinjump(coin_jump)
            explaining = get_predictions(extracted_state, nsfr)
            action = explaining_to_action(explaining)

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

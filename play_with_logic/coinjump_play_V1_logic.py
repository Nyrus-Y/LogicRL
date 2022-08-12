import random
import time

import numpy as np
from src.coinjump.imageviewer import ImageViewer

from src.CoinJump.coinjump.coinjump import ParameterizedLevelGenerator_V1
from src.CoinJump.coinjump.coinjump import CoinJump
from src.util import extract_for_explaining, explaining_nsfr, action_select

KEY_SPACE = 32
# KEY_SPACE = 32
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

    coin_jump = CoinJump(start_on_first_action=True, V1=True)
    # level_generator = DummyGenerator()

    # change generator to choose env
    level_generator = ParameterizedLevelGenerator_V1()

    level_generator.generate(coin_jump, seed=seed)
    # level_generator.generate(coin_jump, seed=seed)
    coin_jump.render()

    return coin_jump


def run():
    coin_jump = create_coinjump_instance()
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

        # step game
        action = []

        if KEY_r in viewer.pressed_keys:
            coin_jump = create_coinjump_instance()
            print("--------------------------     next game    --------------------------")

        if not coin_jump.level.terminated:

            # extract state for explaining
            prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
                         'right_go_to_door']
            extracted_state = extract_for_explaining(coin_jump)
            explaining = explaining_nsfr(extracted_state, 'coinjump1', prednames)
            # explaining = explaining_nsfr_combine(extracted_state,'coinjump_D','coinjump_KD')
            action = action_select(explaining)

            if last_explaining is None:
                print(explaining)
                last_explaining = explaining
            elif explaining != last_explaining:
                print(explaining)
                last_explaining = explaining

        # else:
        #     if KEY_a in viewer.pressed_keys:
        #         action.append(CoinJumpActions.MOVE_LEFT)
        #     if KEY_d in viewer.pressed_keys:
        #         action.append(CoinJumpActions.MOVE_RIGHT)
        #     if (KEY_SPACE in viewer.pressed_keys) or (KEY_w in viewer.pressed_keys):
        #         action.append(CoinJumpActions.MOVE_UP)
        #     if KEY_s in viewer.pressed_keys:
        #         action.append(CoinJumpActions.MOVE_DOWN)

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

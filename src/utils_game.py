import random
import time
from argparse import ArgumentParser
import pathlib
import torch
import numpy as np
import gym
import os
import gym3

from environments.procgen.utils_bf import *
from environments.coinjump.utils_cj import *
from environments.procgen.procgen import ProcgenGym3Env
from environments.coinjump.coinjump.coinjump.actions import coin_jump_actions_from_unified
from environments.coinjump.coinjump.imageviewer import ImageViewer
from environments.coinjump.utils_cj import extract_for_explaining
from environments.coinjump.coinjump_learn.training.data_transform import extract_state, sample_to_model_input_V1, \
    collate
from nsfr.utils import show_explaining
from environments.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from environments.coinjump.coinjump.coinjump.coinjump import CoinJump


def render_coinjump(model, args):
    def setup_image_viewer(coinjump):
        viewer = ImageViewer(
            "coinjump",
            coinjump.camera.height,
            coinjump.camera.width,
            monitor_keyboard=True,
            # relevant_keys=set('W','A','S','D','SPACE')
        )
        return viewer

    def create_coinjump_instance(seed=None):
        seed = random.randint(0, 100000000)

        # level_generator = DummyGenerator()

        coin_jump = CoinJump(V1=True)
        level_generator = ParameterizedLevelGenerator_V1()
        level_generator.generate(coin_jump, seed=seed)
        coin_jump.render()

        return coin_jump

    seed = random.randint(0, 100000000)
    print(seed)
    coin_jump = create_coinjump_instance(seed=seed)
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    num_epi = 0
    max_epi = 100
    total_reward = 0
    last_explaining = None
    epi_reward = 0
    if args.alg == 'logic':
        prednames = model.get_prednames()
    # while True:
    while num_epi <= max_epi:
        # control framerate
        current_frame_time = time.time()
        # limit frame rate
        if last_frame_time + target_frame_duration > current_frame_time:
            sl = (last_frame_time + target_frame_duration) - current_frame_time
            time.sleep(sl)
            continue
        last_frame_time = current_frame_time  # save frame start time for next iteration
        # step game
        if not coin_jump.level.terminated:
            if args.alg == 'logic':
                # extract state for explaining
                extracted_state = extract_for_explaining(coin_jump)
                prediction = model(extracted_state)
                action = prediction_to_action_cj(prediction, args, prednames)
                print(show_explaining(prediction, prednames))
                action = coin_jump_actions_from_unified(action)
            elif args.alg == 'ppo':
                model_input = sample_to_model_input_V1((extract_state(coin_jump), []))
                model_input = collate([model_input])
                model_input = model_input['state']
                prediction = model(model_input)
                action = coin_jump_actions_from_unified(torch.argmax(prediction).cpu().item() + 1)
            elif args.alg == 'random':
                action = coin_jump_actions_from_unified(random.randint(0, 10))
            # action = agent.act(coin_jump.state)
        else:
            coin_jump = create_coinjump_instance(seed=seed)
            print("epi_reward: ", round(epi_reward, 2))
            print("--------------------------     next game    --------------------------")
            total_reward += epi_reward
            epi_reward = 0
            action = []
            num_epi += 1

        reward = coin_jump.step(action)
        # print(reward)
        epi_reward += reward
        np_img = np.asarray(coin_jump.camera.screen)
        viewer.show(np_img[:, :, :3])

        # terminated = coin_jump.level.terminated
        # if terminated:
        #    break
        if viewer.is_escape_pressed:
            break

    # print("average episode reward: ", total_reward / max_epi)
    print("Terminated")


def render_bigfish(model, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    # env_name = "bigfishm"
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    reward, obs, done = env.observe()
    state = extract_state_bf(obs['positions'], args)

    # print(model.state_dict())
    while True:
        # select action with policy
        prediction = model(state)
        prediction = torch.argmax(prediction)
        action = prediction_to_action_bf(prediction, args, model)
        env.act(action)
        rew, obs, done = env.observe()
        print(obs["positions"])
        state = extract_state_bf(obs["positions"], args)


def render_heist(model, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    # env_name = "bigfishm"
    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    env = gym3.ViewerWrapper(env, info_key="rgb")

    reward, obs, done = env.observe()
    # print(obs["positions"])

    # state = extract_state_bf(obs['positions'], args)

    # print(model.state_dict())
    i = 0
    while True:
        # select action with policy
        # prediction = model(state)
        # prediction = torch.argmax(prediction)
        # action = prediction_to_action_bf(prediction, args, model)
        action = np.random.randint([9])
        env.act(action)
        rew, obs, done = env.observe()
        if i % 40 == 0:
            print("\n"*50)
            print(obs["positions"])
        i += 1
        # state = extract_state_bf(obs["positions"], args)

import random
import time
import gym3
import numpy as np
import os
from environments.procgen.procgen import ProcgenGym3Env
from environments.coinjump.coinjump.imageviewer import ImageViewer
from environments.coinjump.coinjump.coinjump.paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1
from environments.coinjump.coinjump.coinjump.coinjump import CoinJump


def render_coinjump(agent, args):
    def setup_image_viewer(coinjump):
        viewer = ImageViewer(
            "coinjump",
            coinjump.camera.height,
            coinjump.camera.width,
            monitor_keyboard=True,
        )
        return viewer

    def create_coinjump_instance(seed=None):

        # level_generator = DummyGenerator()
        coin_jump = CoinJump()
        level_generator = ParameterizedLevelGenerator_V1()
        level_generator.generate(coin_jump, seed=seed)
        coin_jump.render()

        return coin_jump

    # seed = random.randint(0, 100000000)
    # print(seed)
    coin_jump = create_coinjump_instance()
    viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    num_epi = 0
    max_epi = 100
    total_reward = 0
    epi_reward = 0

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
            action = agent.act(coin_jump)
        else:
            coin_jump = create_coinjump_instance()
            print("epi_reward: ", round(epi_reward, 2))
            print("--------------------------     next game    --------------------------")
            total_reward += epi_reward
            epi_reward = 0
            action = 0
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


def render_bigfish(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        while True:
            print(obs['positions'])
            action = agent.act(obs['positions'])
            env.act(action)
            rew, obs, done = env.observe()


def render_heist(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        i = 0
        while True:
            action = agent.act(obs)
            env.act(action)
            rew, obs, done = env.observe()
            i += 1


def render_ecoinrun(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        i = 0
        while True:
            action = agent.act(obs)
            env.act(action)
            rew, obs, done = env.observe()
            # if i % 40 == 0:
            #     print("\n" * 50)
            #     print(obs["positions"])
            i += 1

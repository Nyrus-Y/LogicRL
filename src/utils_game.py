import ast
import csv
import io
import os
import random
import sys
import time

import gym3
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from environments.coinjump.coinjump.coinjump.actions import CoinJumpActions
from environments.coinjump.coinjump.coinjump.coinjump import CoinJump
from environments.coinjump.coinjump.coinjump.paramLevelGenerator import \
    ParameterizedLevelGenerator
from environments.coinjump.coinjump.imageviewer import ImageViewer
from environments.procgen.procgen import ProcgenGym3Env


def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def run(env, nb_games=20):
    """
    Display a window to the user and loop until the window is closed
    by the user.
    """
    prev_time = env._renderer.get_time()
    env._renderer.start()
    env._draw()
    env._renderer.finish()

    old_stdout = sys.stdout  # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    # whatWasPrinted = buffer.getvalue()  # Return a str containing the entire contents of the buffer.
    while buffer.getvalue().count("final info") < nb_games:
        now = env._renderer.get_time()
        dt = now - prev_time
        prev_time = now
        if dt < env._sec_per_timestep:
            sleep_time = env._sec_per_timestep - dt
            time.sleep(sleep_time)

        keys_clicked, keys_pressed = env._renderer.start()
        if "O" in keys_clicked:
            env._overlay_enabled = not env._overlay_enabled
        env._update(dt, keys_clicked, keys_pressed)
        env._draw()
        env._renderer.finish()
        if not env._renderer.is_open:
            break
    sys.stdout = old_stdout  # Put the old stream back in place
    all_summaries = [line for line in buffer.getvalue().split("\n") if line.startswith("final")]
    return all_summaries


def get_values(summaries, key_str, stype=float):
    all_values = []
    for line in summaries:
        dico = ast.literal_eval(line[11:])
        all_values.append(stype(dico[key_str]))
    return all_values


def render_getout(agent, args):
    KEY_SPACE = 32
    # KEY_SPACE = 32
    KEY_w = 119
    KEY_a = 97
    KEY_s = 115
    KEY_d = 100
    KEY_r = 114

    def setup_image_viewer(coinjump):
        viewer = ImageViewer(
            "coinjump",
            coinjump.camera.height,
            coinjump.camera.width,
            monitor_keyboard=True,
        )
        return viewer

    def create_coinjump_instance(args, seed=None):
        if args.env == 'getoutplus':
            enemies = True
        else:
            enemies = False
        # level_generator = DummyGenerator()
        coin_jump = CoinJump()
        level_generator = ParameterizedLevelGenerator(enemies=enemies)
        level_generator.generate(coin_jump, seed=seed)
        coin_jump.render()

        return coin_jump

    # seed = random.randint(0, 100000000)
    # print(seed)
    coin_jump = create_coinjump_instance(args)
    # viewer = setup_image_viewer(coin_jump)

    # frame rate limiting
    fps = 10
    target_frame_duration = 1 / fps
    last_frame_time = 0

    num_epi = 1
    max_epi = 5
    total_reward = 0
    epi_reward = 0
    current_reward = 0
    step = 0
    last_explaining = None

    # print weighted action rules
    agent.model.print_program()

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
        elif args.alg == 'human':
            head = ['episode', 'reward']
        writer.writerow(head)

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
        step += 1
        action = []
        if not coin_jump.level.terminated:
            if args.alg == 'logic':
                action, explaining, atom_grad = agent.act(coin_jump)
            elif args.alg == 'ppo':
                action = agent.act(coin_jump)
            elif args.alg == 'human':
                if KEY_a in viewer.pressed_keys:
                    action.append(CoinJumpActions.MOVE_LEFT)
                if KEY_d in viewer.pressed_keys:
                    action.append(CoinJumpActions.MOVE_RIGHT)
                if (KEY_SPACE in viewer.pressed_keys) or (KEY_w in viewer.pressed_keys):
                    action.append(CoinJumpActions.MOVE_UP)
                if KEY_s in viewer.pressed_keys:
                    action.append(CoinJumpActions.MOVE_DOWN)
        else:
            coin_jump = create_coinjump_instance(args)
            # print("epi_reward: ", round(epi_reward, 2))
            print("--------------------------     next game    --------------------------")
            if args.alg == 'human':
                data = [(num_epi, round(epi_reward, 2))]
                writer.writerows(data)
            total_reward += epi_reward
            epi_reward = 0
            action = 0
            # average_reward = round(total_reward / num_epi, 2)
            num_epi += 1
            step = 0

        reward = coin_jump.step(action)
        score = coin_jump.get_score()
        current_reward += reward
        average_reward = round(current_reward / num_epi, 2)
        if args.alg == 'logic':
            if last_explaining is None:
                print(explaining)
                last_explaining = explaining
            elif explaining != last_explaining:
                print(explaining)
                last_explaining = explaining

        if args.log:
            if args.alg == 'logic':
                probs = agent.get_probs()
                logic_state = agent.get_state(coin_jump)
                data = [(num_epi, step, reward, average_reward, logic_state, probs)]
                writer.writerows(data)
            elif args.alg == 'ppo' or args.alg == 'random':
                data = [(num_epi, step, reward, average_reward)]
                writer.writerows(data)
        # print(reward)
        epi_reward += reward

        if args.render:
            action_preds = agent.model.prednames
            atoms = [x for x in agent.model.atoms]
            action_atoms = [x for x in agent.model.atoms if x.pred.name in action_preds]
            # action_preds = ['jump', 'left_to_get_key', 'right_to_get_key', 'left_to_door', 'right_to_door']
            if step % 1 == 0:
                folder_path = f'grad_plot/{args.rules}/episode_{num_epi}'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                    print(f'folder {folder_path} created!!!!')
                env_img = np.asarray(coin_jump.camera.screen)
                Image.fromarray(env_img).save(f"{folder_path}/{args.rules}_{step}_env.png")
                # grad_img = np.round(agent.action_grad, 2)
                # grad_img = np.log(agent.action_grad+0.01)
                # grad_img = zscore(agent.action_grad)
                grad_img = np.round(agent.action_grad, 2)
                grad_img.astype(np.float32)
                #print("Action Gradients: ", agent.action_grad, agent.action_grad.shape)
                # save grad plot

                # im = ax.imshow(grad_img, cmap="plasma")
                # im = ax.imshow(grad_img, cmap="cividis")

                # prone redundant columns
                # pruned_atoms = atoms
                columns = []
                pruned_atoms = []
                for i in range(len(atoms)):
                    #if True:
                    if i > 1 and grad_img[:,i].max() > 0 and not atoms[i].pred.name in action_preds:
                        columns.append(grad_img[:,i])
                        pruned_atoms.append(atoms[i])
                #grad_img = np.array(columns).reshape((len(action_preds), len(pruned_atoms)))
                grad_img = np.stack(columns, axis=-1)

                fig, ax = plt.subplots(figsize=(6, 6))
                # fig, ax = plt.subplots(figsize=(16, 8))
                # vmax = 0.3
                # vmax = grad_img.max() / 2
                vmax = grad_img.max()
                normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
                # im = ax.imshow(grad_img, cmap="jet", norm=normalize)
                # im = ax.imshow(grad_img, cmap="cividis", norm=normalize)
                im = ax.imshow(grad_img, cmap="cividis")

                # We want to show all ticks...
                ax.set_xticks(np.arange(len(pruned_atoms)))
                ax.set_yticks(np.arange(len(action_preds)))
                plt.yticks(fontname = "monospace", fontsize=12)
                plt.xticks(fontname = "monospace", fontsize=11)
                ax.set_xticklabels([str(x).replace('key(img)','key(agent)') for x in pruned_atoms])
                ax.set_yticklabels([x.replace('_go_get_', '_').replace('_go_to_','_') for x in action_preds])
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                plt.rcParams.update({'font.size': 11})
                for i in range(len(action_preds)):
                    for j in range(len(pruned_atoms)):
                        if grad_img[i, j] > 0:
                            #print(grad_img[i,j], grad_img[i, j] / vmax )
                            if grad_img[i, j] / vmax < 0.8:
                                text = ax.text(j, i, grad_img[i, j], ha="center", va="center", color="w") 
                            else:
                                text = ax.text(j, i, grad_img[i, j], ha="center", va="center", color="b") 

                ax.set_title("Gradients of actions w.r.t. input atoms", fontsize=16)
                fig.tight_layout()
                plt.show()
                plt.savefig(f"{folder_path}/{args.rules}_{step}_action_grad.png")

                # save action-atom dist
                action_atom_dist_img = np.round(agent.action_atom_dist, 2)
                # print(action_atom_dist_img)
                fig, ax = plt.subplots()
                plt.yticks(fontsize=14)
                plt.xticks(fontname = "monospace", fontsize=14)
                # plt.rcParams.update({'font.size': 22})
                #m = ax.imshow(action_atom_dist_img, cmap="YlGn")
                ax.bar(action_preds, action_atom_dist_img, color='darkviolet')

                ax.set_xticks(np.arange(len(action_preds)))
                #ax.set_yticks(np.arange(1))
                # ax.set_xticklabels(action_preds)
                ax.set_xticklabels([x.replace('_go_get_', '_').replace('_go_to_','_') for x in action_preds])
                #ax.set_yticklabels()
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                # Loop over data dimensions and create text annotations.
                #for j in range(len(action_preds)):
                #    text = ax.text(j, 0, action_atom_dist_img[j], ha="center", va="center", color="w") 
                ax.set_title("Distribution over action atoms", fontsize=16)
                fig.tight_layout()
                plt.show()
                plt.savefig(f"{folder_path}/{args.rules}_{step}_action_atom_dist.png")


                #plt.figure(figsize=(16,8))
                #plt.imshow(grad_img)
                #plt.colorbar()
                # plt.imsave(f"grad_plot/{args.rules}_{step}_action_grad.png", grad_img)
                #Image.fromarray(grad_img).save(f"grad_plot/{args.rules}_{step}_action_grad.png")
                #atoms, action_grad = agent.model.get_action_grad()
                #env_image = env.render(mode="rgb_array")
                #print(env_image)
                #old_action = action
            # viewer.show(np_img[:, :, :3])

        # terminated = coin_jump.level.terminated
        # if terminated:
        #    break
        #if viewer.is_escape_pressed:
        #    break

        if coin_jump.level.terminated:
            step = 0
            print(num_epi)
            print('reward: ' + str(round(score, 2)))
        if num_epi > 100:
            break

    print('total reward= ', total_reward)
    print('average reward= ', total_reward / 100)
    # print("average episode reward: ", total_reward / max_epi)
    print("Terminated")


def render_bigfish(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":

        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        all_summaries = run(ia)

        df_scores = get_values(all_summaries, "episode_return")
        data = {'reward': df_scores}
        # convert list to df_scores
        # pd.to_csv(df_scores, f"{player_name}_scores.csv")
        df = pd.DataFrame(data)
        df.to_csv(args.logfile, index=False)

    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        total_r = 0
        epi = 1
        epi_r = 0
        step = 0
        current_r = 0
        last_explaining = None
        while True:
            # print(obs['positions'])
            if args.alg == 'logic':
                action, explaining = agent.act(obs)
            else:
                action = agent.act(obs)
            env.act(action)
            rew, obs, done = env.observe()
            total_r += rew[0]
            epi_r += rew[0]
            current_r += rew[0]
            step += 1
            average_r = round(current_r / epi, 2)
            if args.alg == 'logic':
                if last_explaining is None:
                    print(explaining)
                    last_explaining = explaining
                elif explaining != last_explaining:
                    print(explaining)
                    last_explaining = explaining

            if args.log:
                if args.alg == 'logic':
                    probs = agent.get_probs()
                    logic_state = agent.get_state(obs)
                    data = [(epi, step, rew[0], average_r, logic_state, probs)]
                    writer.writerows(data)
                else:
                    data = [(epi, step, rew[0], average_r)]
                    writer.writerows(data)

            if done:
                epi += 1
                step = 0
                print("episode: ", epi)
                print("reward: ", epi_r)
                epi_r = 0

            if epi > 100:
                break

        print('total reward= ', total_r)
        print('average reward= ', total_r / 100)


def render_heist(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")

    if args.log:
        log_f = open(args.logfile, "w+")
        writer = csv.writer(log_f)

        if args.alg == 'logic':
            head = ['episode', 'step', 'reward', 'average_reward', 'logic_state', 'probs']
            writer.writerow(head)
        elif args.alg == 'ppo' or args.alg == 'random':
            head = ['episode', 'step', 'reward', 'average_reward']
            writer.writerow(head)

    if agent == "human":

        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        all_summaries = run(ia)

        df_scores = get_values(all_summaries, "episode_return")
        data = {'reward': df_scores}
        # convert list to df_scores
        # pd.to_csv(df_scores, f"{player_name}_scores.csv")
        df = pd.DataFrame(data)
        df.to_csv(args.logfile, index=False)
    else:
        if args.render:
            env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        step = 0
        total_r = 0
        epi_r = 0
        epi = 1
        current_r = 0
        last_explaining = None
        while True:
            step += 1
            if args.alg == 'logic':
                action, explaining = agent.act(obs)
            else:
                action = agent.act(obs)

            env.act(action)
            rew, obs, done = env.observe()
            total_r += rew[0]
            epi_r += rew[0]
            current_r += rew[0]
            average_r = round(current_r / epi, 2)
            if args.alg == 'logic':
                if last_explaining is None:
                    print(explaining)
                    last_explaining = explaining
                elif explaining != last_explaining:
                    print(explaining)
                    last_explaining = explaining

            if args.log:
                if args.alg == 'logic':
                    probs = agent.get_probs()
                    logic_state = agent.get_state(obs)
                    data = [(epi, step, rew[0], average_r, logic_state, probs)]
                    writer.writerows(data)
                else:
                    data = [(epi, step, rew[0], average_r)]
                    writer.writerows(data)

            if done:
                epi += 1
                step = 0
                print("episode: ", epi)
                print("reward: ", epi_r)
                epi_r = 0
            if epi > 100:
                break

        print('total reward= ', total_r)
        print('average reward= ', total_r / 100)


def render_ecoinrun(agent, args):
    seed = random.seed() if args.seed is None else int(args.seed)
    os.mkdir('grad_plot/')

    env = ProcgenGym3Env(num=1, env_name=args.env, render_mode="rgb_array")
    if agent == "human":
        ia = gym3.Interactive(env, info_key="rgb", height=768, width=768)
        ia.run()
    else:
        #env = gym3.ViewerWrapper(env, info_key="rgb")
        reward, obs, done = env.observe()
        i = 0
        old_action = None
        while True:
            action = agent.act(obs)
            if action != old_action:
                atoms, action_grad = agent.get_action_grad()
                #print('action: ', action)
                #print('action_grad: ', action_grad)
                env_image = env.render(mode="rgb_array")
                #print(env_image)
                Image.fromarray(env_image).save(f"grad_plot/{args.mode}_{i}.png")
                old_action = action
            env.act(action)
            rew, obs, done = env.observe()
            # if i % 40 == 0:
            #     print("\n" * 50)
            #     print(obs["positions"])
            i += 1

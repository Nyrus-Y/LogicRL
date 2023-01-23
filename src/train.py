import argparse
import os
import time
import gym
import sys
import pickle
import csv
import copy
import numpy as np

# import wandb
import environments.coinjump.env

sys.path.insert(0, '../')

from agents.logic_agent import LogicPPO
from agents.neural_agent import NeuralPPO
from environments.procgen.procgen import ProcgenGym3Env
from utils import make_deterministic, initialize_game, env_step
from config import *
from tqdm import tqdm
from rtpt import RTPT
from make_graph import plot_weights_beam, plot_weights


def main():
    ################### args definition ###################
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                        required=False, action="store", dest="seed", type=int, default=0)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['getout', 'bigfish', 'heist'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['getout', 'bigfish', 'heist'])
    parser.add_argument("-r", "--rules", dest="rules", default=None, required=False,
                        choices=['getout_human_assisted', 'getout_redundant_actions', 'getout_bs_top10',
                                 'getout_bs_top1', 'getout_bs_top3', 'ppo_simple_policy',
                                 'bigfish_human_assisted', 'bigfishcolor', 'bigfish_bs_top5', 'bigfish_bs_top3',
                                 'bigfish_bs_top1', 'bigfish_redundant_actions',
                                 'heist_human_assisted', 'heist_bs_top5', 'heist_bs_top3', 'heist_bs_top1',
                                 'heist_redundant_actions'])
    parser.add_argument('-p', '--plot', help="plot the image of weights", type=bool, default=False, dest='plot')
    parser.add_argument('--recover', help='recover from crash', default=False, type=bool, dest='recover')
    #arg = ['-alg', 'logic', '-m', 'bigfish', '-env', 'bigfish', '-r', 'bigfish_bs_top1']
    args = parser.parse_args()

    #####################################################
    # load environment
    print("training environment name : " + args.env)
    make_deterministic(args.seed)

    #####################################################
    # config setting
    if args.alg == 'ppo':
        update_timestep = max_ep_len * 4
    else:
        update_timestep = max_ep_len * 2

    if args.m == 'heist' and args.alg == 'ppo':
        max_training_timesteps = 5000000
    else:
        max_training_timesteps = 800000
    #####################################################

    if args.m == "getout":
        env = gym.make(args.env, generator_args={"spawn_all_entities": False})
    elif args.m == "bigfish" or args.m == 'heist':
        env = ProcgenGym3Env(num=1, env_name=args.env, render_mode=None)

    #####################################################
    # config = {
    #     "seed": args.seed,
    #     "learning_rate_actor": lr_actor,
    #     "learning_rate_critic": lr_critic,
    #     "epochs": K_epochs,
    #     "gamma": gamma,
    #     "eps_clip": eps_clip,
    #     "max_steps": max_training_timesteps,
    #     "eps start": 1.0,
    #     "eps end": 0.02,
    #     "max_ep_len": max_ep_len,
    #     "update_freq": max_ep_len * 2,
    #     "save_freq": max_ep_len * 50,
    # }
    # if args.rules is not None:
    #     runs_name = str(args.rules) + '_seed_' + str(args.seed)
    # else:
    #     runs_name = str(args.m) + '_' + args.alg + '_seed_' + str(args.seed)

    # wandb.init(project="GETOUT-BS", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="HEIST", entity="nyrus", config=config, name=runs_name)
    # wandb.init(project="BIGFISH", entity="nyrus", config=config, name=runs_name)

    ################### checkpointing ###################

    directory = "checkpoints"
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.rules is not None:
        directory = directory + '/' + args.m + '/' + args.alg + '/' + args.env + '/' + args.rules + '/' + str(
            args.seed) + '/'
    else:
        directory = directory + '/' + args.m + '/' + args.alg + '/' + args.env + '/' + str(args.seed) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # if not args.recover:

    checkpoint_path = directory + "{}_{}.pth".format(args.env, 0)

    print("save checkpoint path : " + checkpoint_path)

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    # print("state space dimension : ", state_dim)
    # print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")

    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    #
    # initialize agent
    if args.alg == "ppo":
        agent = NeuralPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args)
    elif args.alg == "logic":
        agent = LogicPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args)

    if args.recover:
        step_list, reward_list, weights_list = agent.load(checkpoint_path, directory)
    else:
        step_list = []
        reward_list = []
        weights_list = []

    # track total training time
    start_time = time.time()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    image_directory = "image"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    image_directory = image_directory + '/' + args.env + '/'
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    if args.plot:
        if args.alg == 'logic':
            plot_weights_beam(agent.get_weights(), image_directory)
        elif args.alg == 'ppo':
            plot_weights(agent.get_weights(), image_directory)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0

    rtpt = RTPT(name_initials='QD', experiment_name='LogicRL',
                max_iterations=max_training_timesteps)

    # Start the RTPT tracking
    rtpt.start()
    # training loop
    pbar = tqdm(total=max_training_timesteps)
    while time_step <= max_training_timesteps:
        #  initialize game
        state = initialize_game(env, args)
        current_ep_reward = 0

        epsilon = epsilon_func(i_episode)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = agent.select_action(state, epsilon=epsilon)
            reward, state, done = env_step(action, env, args)

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
            pbar.update(1)
            rtpt.step()
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                agent.update()

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                # wandb.log({'reward': print_avg_reward}, step=time_step)
                print_running_reward = 0
                print_running_episodes = 0

                step_list.append([time_step])
                reward_list.append([print_avg_reward])
                weights_list.append([(agent.get_weights().tolist())])

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + "{}_{}_step_{}.pth".format(args.alg, args.env,
                                                                         time_step)
                print("saving model at : " + checkpoint_path)
                agent.save(checkpoint_path, directory, step_list, reward_list, weights_list)
                print("model saved")
                print("Elapsed Time  : ", time.time() - start_time)
                print("--------------------------------------------------------------------------------------------")

                # save image of weights
                if args.plot:
                    if args.alg == 'logic':
                        plot_weights_beam(agent.get_weights(), image_directory, time_step)
                    elif args.alg == 'ppo':
                        plot_weights(agent.get_weights(), image_directory, time_step)
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()

    # print total training time
    print("============================================================================================")
    with open(directory + '/' + 'data.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        header = ('steps', 'reward')
        dataset.writerow(header)
        data = np.hstack((step_list, reward_list))
        for row in data:
            dataset.writerow(row)

    with open(directory + '/' + 'weights.csv', 'w', newline='') as f:
        dataset = csv.writer(f)
        for row in weights_list:
            dataset.writerow(row)

    end_time = time.time()
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()

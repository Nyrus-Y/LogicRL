import argparse
import os
import time
import gym
import environments.coinjump.coinjump_learn.env
from agents.logic_agent import LogicPPO
from agents.neural_agent import NeuralPPO
from environments.procgen.utils_bf import *
from environments.coinjump.utils_cj import *
from environments.procgen.procgen import ProcgenGym3Env
from utils import make_deterministic
from config import *


def main():
    ################### args definition ###################
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                        required=True, action="store", dest="seed", type=int, default=0)
    parser.add_argument("-alg", "--algorithm", help="algorithm that to use",
                        action="store", dest="alg", required=True,
                        choices=['ppo', 'logic'])
    parser.add_argument("-m", "--mode", help="the game mode you want to play with",
                        required=True, action="store", dest="m",
                        choices=['coinjump', 'bigfish', 'heist'])
    # parser.add_argument("-env", "--environment", help="Game to play with",
    #                     required=True, action="store", dest="env",
    #                     choices=['coinjump', 'bigfish', 'heist'])
    parser.add_argument("-env", "--environment", help="environment of game to use",
                        required=True, action="store", dest="env",
                        choices=['CoinJumpEnvLogic-v0', 'CoinJumpEnvNeural-v0',
                                 'bigfishm', 'bigfishc', 'heist'])
    parser.add_argument("-r", "--rules", dest="rules", default=None,
                        required=False,
                        choices=['coinjump_5a', 'bigfish_simplified_actions', 'heist', 'ppo_simple_policy'])
    parser.add_argument("--recover", help="Recover from the last trained agent",
                        action="store_true", dest="recover", default=False)
    parser.add_argument("--load", help="Pytorch file to load if continue training",
                        action="store_true", dest="load", default=False)

    #args = ['-s', 0, '-m', 'bigfish', '-alg', 'logic', '-env', 'bigfishm', '-r', 'bigfish_simplified_actions']
    #args = parser.parse_args(args)
    args = parser.parse_args()

    #####################################################
    # load environment

    if args.m == "coinjump":
        env = gym.make(args.env, generator_args={"spawn_all_entities": False})
    elif args.m == "bigfish" or args.m == 'heist':
        env = ProcgenGym3Env(num=1, env_name=args.env, render_mode=None)

    print("training environment name : " + args.env)
    make_deterministic(args.seed)
    #####################################################

    ################### checkpointing ###################

    directory = "checkpoints"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + args.m + '/' + args.alg + '/' + args.env + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "{}_{}_{}.pth".format(args.env, args.seed, 0)
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
        prednames = agent.get_prednames()

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

    # checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, 0)

    # plot_weights(agent.get_weights(), image_directory)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        #  initialize game
        if args.m == 'bigfish':
            reward, obs, done = env.observe()
            state = extract_state_bf(obs['positions'], args)
        elif args.m == 'coinjump':
            state = env.reset()

        current_ep_reward = 0

        epsilon = epsilon_func(i_episode)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            action = agent.select_action(state, epsilon=epsilon)

            # steps of each game
            if args.m == 'bigfish':
                if args.alg == 'ppo':
                    action = prediction_to_action_bf(action, args)
                elif args.alg == 'logic':
                    action = prediction_to_action_bf(action, args, prednames)
                env.act(action)
                rew, obs, done = env.observe()
                state = extract_state_bf(obs["positions"], args)
                reward = rew[0]

            elif args.m == 'coinjump':
                if args.alg == 'ppo':
                    action = prediction_to_action_cj(action, args)
                elif args.alg == 'logic':
                    action = prediction_to_action_cj(action, args, prednames)
                state, reward, done, _ = env.step(action)
                if args.rules == 'ppo_simple_policy':
                    # simpler policy
                    if action in [3]:
                        reward -= 0.2

            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)

            time_step += 1
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

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + "{}_{}_seed_{}_epi_{}.pth".format(args.alg, args.env, args.seed,
                                                                                i_episode)
                print("saving model at : " + checkpoint_path)
                agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", time.time() - start_time)
                print("--------------------------------------------------------------------------------------------")

                # save image of weights
                # plot_weights(agent.get_weights(), image_directory, time_step)
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = time.time()
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    main()

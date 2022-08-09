import math
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
import os

import gym
import time
import numpy as np
import src.coinjump_learn.env

from src.valuation import RLValuationModule
from src.facts_converter import FactsConverter
from src.logic_utils import build_infer_module, get_lang
from src.util import num_action_select
from src.nsfr_training import NSFReasoner
from src.coinjump_learn.models.mlpCriticController import MLPCriticController

device = torch.device('cuda:0')


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]


class NSFR_ActorCritic(nn.Module):
    def __init__(self, rng=None):
        super(NSFR_ActorCritic, self).__init__()

        self.rng = random.Random() if rng is None else rng
        self.num_actions = 6
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device="cuda"))

        self.actor = self.get_nsfr_model()
        self.critic = MLPCriticController(out_size=1)

    def forward(self):
        raise NotImplementedError

    def act(self, state, epsilon=0.0):

        action_probs = self.actor(state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
        else:
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    def get_nsfr_model(self):

        lark_path = 'E:\\Github\\Use Knowledge Representation and Reasoning for the policy\\src\\lark\\exp.lark'
        lang_base_path = 'E:\\Github\\Use Knowledge Representation and Reasoning for the policy\\data\\lang\\'

        device = torch.device('cuda:0')
        lang, clauses, bk, atoms = get_lang(
            lark_path, lang_base_path, 'coinjump', 'coinjump')

        PM = None
        VM = RLValuationModule(lang=lang, device=device)
        FC = FactsConverter(lang=lang, perception_module=PM, valuation_module=VM, device=device)
        # TODO change m

        m = len(clauses)
        # m = 6
        IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=True, device=device)
        # Neuro-Symbolic Forward Reasoner
        NSFR = NSFReasoner(perception_module=PM, facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk,
                           clauses=clauses, train=True)
        return NSFR


class PPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.policy = NSFR_ActorCritic().to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = NSFR_ActorCritic().to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, epsilon=0.0):
        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, epsilon=epsilon)

        self.buffer.states.append(state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        # torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(self.policy_old, checkpoint_path)

    def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_old = torch.load(checkpoint_path)
        self.policy = torch.load(checkpoint_path)

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction


# ===== ACTUAL TRAINING HERE =======

def main():
    ####### initialize environment hyperparameters ######

    random_seed = 123456  # set random seed if required (0 = no random seed)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        # torch.use_deterministic_algorithms(True)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # TODO choose env
    # env_name = "CoinJumpEnv-v0"
    # env_name = "CoinJumpEnvKD-v0"
    # env_name = "CoinJumpEnvDodge-v0"
    env_name = "CoinJumpEnv-v2"

    # max_ep_len = 1000  # max timesteps in one episode
    max_ep_len = 500  # max timesteps in one episode
    # max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps
    max_training_timesteps = 100000000  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5  # log avg reward in the interval (in num timesteps)
    # save_model_freq = int(2e4)  # save model frequency (in num timesteps)
    # save_model_freq = 500000  # save model frequency (in num timesteps)
    save_model_freq = 50000  # save model frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################

    # update_timestep = max_ep_len * 4  # update policy every n episodes
    # update_timestep = max_ep_len * 10  # update policy every n episodes
    update_timestep = max_ep_len * 5  # debug update policy every n episodes
    # update_timestep = 2  # update policy every n episodes
    # K_epochs = 80  # update policy for K epochs (= # update steps)
    # K_epochs = 20
    K_epochs = 20  # update policy for K epochs (= # update steps)
    # eps_clip = 0.2  # clip parameter for PPO
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    # gamma = 0.95
    # gamma = 1.0  # the coinjump1 environment already has a time penality for its reward

    # optimizer = torch.optim.Adam
    # lr_actor = 0.0003  # learning rate for actor network
    # lr_critic = 0.001  # learning rate for critic network
    optimizer = torch.optim.Adam
    # optimizer = torch.optim.SGD
    # lr_actor = 0.001  # learning rate for actor network
    lr_actor = 0.001  # learning rate for actor network
    # lr_critic = 0.0003  # learning rate for critic network
    lr_critic = 0.0003  # learning rate for critic network

    # epsilon = 1.0  # 1.0=completely random, 0.0=pure policy actions
    # epsilon = lambda episode: 0.0
    # epsilon = math.exp(-i_episode/2000) #i=200: 0.90, i=500: 0.77, i=1000: 0.60, i=2000:0.36, i=4000: 0.13, i=6000:0.04, i=10k:0.006
    epsilon_func = lambda episode: math.exp(-episode / 500)
    # epsilon = max(math.exp(-i_episode/2000), 0.02)
    # epsilon = math.exp(-i_episode/2000) + 0.025+math.cos(i_episode*(2*math.pi)/5000)*0.025

    #####################################################

    print("training environment name : " + env_name)

    env = gym.make(env_name, generator_args={"spawn_all_entities": False})

    env.seed(random_seed)

    # state space dimension
    # state_dim = env.observation_space.shape[0]

    # action space dimension
    # action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    #####################################################

    ################### checkpointing ###################

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, 0)
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
    # initialize a PPO agent
    ppo_agent = PPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = time.time()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        epsilon = epsilon_func(i_episode)

        for t in range(1, max_ep_len + 1):

            # select action with policy
            # extract state for explaining
            # prednames = ['jump', 'left_go_get_key', 'right_go_get_key', 'left_go_to_door',
            #              'right_go_to_door', 'stay']

            action = ppo_agent.select_action(state, epsilon=epsilon)
            action = num_action_select(action)
            state, reward, done, _ = env.step(action)

            # simpler policy
            # if action in [3, 5, 6]:
            #     reward -= 0.15
            # elif action == 0:
            #     reward += 0.1

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            # if time_step % log_freq == 0:
            #     # log average reward till last episode
            #     log_avg_reward = log_running_reward / log_running_episodes
            #     log_avg_reward = round(log_avg_reward, 4)
            #
            #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            #     log_f.flush()
            #
            #     log_running_reward = 0
            #     log_running_episodes = 0

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
                checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, i_episode)
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", time.time() - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
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

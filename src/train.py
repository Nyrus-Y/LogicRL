import argparse
import numpy as np
import torch
from environments.bigfish
from config import *
import random


def make_deterministic(seed, environment):
    np.random.seed(seed)
    environment.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set all environment deterministic to seed {seed}")
    random.seed(random_seed)


parser = argparse.ArgumentParser()

parser.add_argument("-alg", "--algo", help="Activation function to use",
                    action="store", dest="algo", required=True,
                    choices=['PPO', 'SimplePPO', 'BeamRules'])
parser.add_argument("-env", "--environment", help="Environment to train on",
                    required=True, action="store", dest="env",
                    choices=['coinjump', 'bigfish', 'heist'])
parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                    required=True, action="store", dest="seed", type=int,
                    default=0)
parser.add_argument("--recover", help="Recover from the last trained agent",
                    action="store_true", dest="recover", default=False)
parser.add_argument("--load", help="Pytorch file to load if continue training",
                    action="store_true", dest="load", default=False)


args = parser.parse_args()


# load environment
if args.env == "coinjump":
    env_name = "CoinJumpEnv-v2"
elif args.env == "bigfish":
    env_name = "bigfishm"
env = ProcgenGym3Env(num=1, env_name=env_name, render_mode=None)
make_deterministic(args.seed, env)

# load algorithm
if args.algo == "nppo":
    agent = NeuralPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip)
elif args.algo == "logic":
    agent = LogicPPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip)


print("training environment name : " + env_name)

env = ProcgenGym3Env(num=1, env_name=env_name, render_mode=None)

#####################################################

################### checkpointing ###################

directory = "logic_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_path = directory + "logic_{}_{}_{}.pth".format(env_name, random_seed, 0)
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
agent = PPO(lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip)

# track total training time
start_time = time.time()
print("Started training at (GMT) : ", start_time)

print("============================================================================================")

image_directory = "image"
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

image_directory = image_directory + '/' + env_name + '/'
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, 0)

plot_weights(agent.get_weights(), image_directory)

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# training loop
while time_step <= max_training_timesteps:

    reward, obs, done = env.observe()
    state = extract_reasoning_state(obs['positions'])
    current_ep_reward = 0

    epsilon = epsilon_func(i_episode)

    for t in range(1, max_ep_len + 1):

        reward = 0
        # select action with policy
        action = agent.select_action(state, epsilon=epsilon)
        # if action in [0, 1, 2, 3]:
        #     reward += 0.001
        action = num_action_select(action)
        # state, reward, done, _ = env.step(action)
        env.act(action)
        rew, obs, done = env.observe()
        state = extract_reasoning_state(obs["positions"])
        reward += rew[0]

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
            checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, i_episode)
            print("saving model at : " + checkpoint_path)
            agent.save(checkpoint_path)
            print("model saved")
            print("Elapsed Time  : ", time.time() - start_time)
            print("--------------------------------------------------------------------------------------------")

            # save image of weights
            plot_weights(agent.get_weights(), image_directory, time_step)
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

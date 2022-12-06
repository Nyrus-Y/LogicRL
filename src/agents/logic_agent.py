import os
import random
import torch
import torch.nn as nn

from torch.distributions import Categorical
from nsfr.utils import get_nsfr_model
from .MLPController.mlpbigfish import MLPBigfish
from .MLPController.mlpcoinjump import MLPCoinjump
from .utils_coinjump import extract_state_coinjump, preds_to_action_coinjump, action_map_coinjump
from .utils_bigfish import extract_state_bigfish, preds_to_action_bigfish, action_map_bigfish
from .utils_heist import extract_state_heist, action_map_heist

device = torch.device('cuda:0')


class NSFR_ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(NSFR_ActorCritic, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = get_nsfr_model(self.args, train=True)
        if self.args.m == 'bigfish' or self.args.m == 'heist':
            self.critic = MLPBigfish(out_size=1, logic=True)
        elif self.args.m == 'coinjump':
            self.critic = MLPCoinjump(out_size=1, logic=True)

        self.prednames = self.get_prednames()
        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device="cuda"))

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

    def get_prednames(self):
        return self.actor.get_prednames()


class LogicPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.args = args
        self.policy = NSFR_ActorCritic(self.args).to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = NSFR_ActorCritic(self.args).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.prednames = self.get_prednames()

    def select_action(self, state, epsilon=0.0):

        # extract state for different games
        if self.args.m == 'coinjump':
            state = extract_state_coinjump(state)
        elif self.args.m == 'bigfish':
            state = extract_state_bigfish(state['positions'], self.args)
        elif self.args.m == 'eheist':
            state = extract_state_heist(state['positions'])

        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, epsilon=epsilon)

        self.buffer.states.append(state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        # different games use different action system, need to map it to the correct action.
        # action of logic game means a String, need to map string to the correct action,
        action = action.item()
        if self.args.m == 'coinjump':
            action = action_map_coinjump(action, self.args, self.prednames)
        elif self.args.m == 'bigfish':
            action = action_map_bigfish(action, self.args, self.prednames)
        elif self.args.m == 'eheist':
            action = action_map_heist(action, self.args, self.prednames)

        return action

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
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        # torch.save(self.policy_old, checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # self.policy_old = torch.load(checkpoint_path)
        # self.policy = torch.load(checkpoint_path)

    def get_predictions(self, state):
        self.prediction = state
        return self.prediction

    def get_weights(self):
        return self.policy.actor.get_params()

    def get_prednames(self):
        return self.policy.actor.get_prednames()


class LogicPlayer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prednames = model.get_prednames()

    def act(self, state):
        # TODO how to do if-else only once?
        if self.args.m == 'coinjump':
            action = self.coinjump_actor(state)
        elif self.args.m == 'bigfish':
            action = self.bigfish_actor(state)
        elif self.args.m == 'heist':
            action = self.heist_actor(state)
        return action

    def coinjump_actor(self, coinjump, show_explaining=False):
        extracted_state = extract_state_coinjump(coinjump)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        if show_explaining:
            print(self.prednames[prediction])
        action = preds_to_action_coinjump(prediction, self.prednames)
        return action

    def bigfish_actor(self, state):
        state = extract_state_bigfish(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        action = preds_to_action_bigfish(action, self.prednames)
        return action

    def heist_actor(self, state):
        pass


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

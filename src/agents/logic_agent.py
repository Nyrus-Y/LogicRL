import os
import random
import torch
import torch.nn as nn

from torch.distributions import Categorical
from src.nsfr.utils import get_nsfr_model
from .MLPController.mlpCriticController import MLPCriticController
from .MLPController.mlpbigfish import MLPBigfish

device = torch.device('cuda:0')


class NSFR_ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(NSFR_ActorCritic, self).__init__()
        self.rng = random.Random() if rng is None else rng

        self.actor = get_nsfr_model(args)
        if args.m == 'bigfish':
            self.critic = MLPBigfish(out_size=1)
        elif args.m == 'coinjump':
            self.critic = MLPCriticController(out_size=1)

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

    # def get_nsfr_model(self, args):
    #     current_path = os.getcwd()
    #     lark_path = os.path.join(current_path, 'lark/exp.lark')
    #     lang_base_path = os.path.join(current_path, 'data/lang/')
    #     # TODO
    #     device = torch.device('cuda:0')
    #     lang, clauses, bk, atoms = get_lang(
    #         lark_path, lang_base_path, args.env, args.rules)
    #
    #     VM = RLValuationModule(lang=lang, device=device)
    #     FC = FactsConverter(lang=lang, valuation_module=VM, device=device)
    #     # m = len(clauses)
    #     m = 5
    #     IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=True, device=device)
    #     # Neuro-Symbolic Forward Reasoner
    #     NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, train=True)
    #     return NSFR


class LogicPPO:
    def __init__(self, lr_actor, lr_critic, optimizer, gamma, K_epochs, eps_clip, args):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.policy = NSFR_ActorCritic(args).to(device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.get_params(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = NSFR_ActorCritic(args).to(device)
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

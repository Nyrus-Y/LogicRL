import os
import random
import torch
import torch.nn as nn
import wandb
from torch.distributions import Categorical
from nsfr.utils import get_nsfr_model
from .MLPController.mlpbigfish import MLPBigfish
from .MLPController.mlpcoinjump import MLPCoinjump
from .MLPController.mlpheist import MLPHeist
from .utils_coinjump import extract_logic_state_coinjump, preds_to_action_coinjump, action_map_coinjump, \
    extract_neural_state_coinjump
from .utils_bigfish import extract_logic_state_bigfish, preds_to_action_bigfish, action_map_bigfish, \
    extract_neural_state_bigfish
from .utils_heist import extract_logic_state_heist, action_map_heist, extract_neural_state_heist, \
    preds_to_action_heist

device = torch.device('cuda:0')


class NSFR_ActorCritic(nn.Module):
    def __init__(self, args, rng=None):
        super(NSFR_ActorCritic, self).__init__()
        self.rng = random.Random() if rng is None else rng
        self.args = args
        self.actor = get_nsfr_model(self.args, train=True)
        self.prednames = self.get_prednames()
        if self.args.m == 'bigfish':
            self.critic = MLPBigfish(out_size=1, logic=True)
        elif self.args.m == 'getout':
            self.critic = MLPCoinjump(out_size=1, logic=True)
        elif self.args.m == 'heist':
            self.critic = MLPHeist(out_size=1, logic=True)
        self.num_actions = len(self.prednames)
        self.uniform = Categorical(
            torch.tensor([1.0 / self.num_actions for _ in range(self.num_actions)], device="cuda"))

    def forward(self):
        raise NotImplementedError

    def act(self, logic_state, epsilon=0.0):

        action_probs = self.actor(logic_state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
            action = dist.sample()
        else:
            dist = Categorical(action_probs)
            action = (action_probs[0] == max(action_probs[0])).nonzero(as_tuple=True)[0].squeeze(0).to(device)
            if torch.numel(action) > 1:
                action = action[0]
        # action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, neural_state, logic_state, action):
        action_probs = self.actor(logic_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(neural_state)

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
        if self.args.m == 'getout':
            logic_state = extract_logic_state_coinjump(state, self.args)
            neural_state = extract_neural_state_coinjump(state, self.args)
        elif self.args.m == 'bigfish':
            logic_state = extract_logic_state_bigfish(state, self.args)
            neural_state = extract_neural_state_bigfish(state, self.args)
        elif self.args.m == 'heist':
            logic_state = extract_logic_state_heist(state, self.args)
            neural_state = extract_neural_state_heist(state, self.args)

        # select random action with epsilon probability and policy probiability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(logic_state, epsilon=epsilon)

        self.buffer.neural_states.append(neural_state)
        self.buffer.logic_states.append(logic_state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        # different games use different action system, need to map it to the correct action.
        # action of logic game means a String, need to map string to the correct action,
        action = action.item()
        if self.args.m == 'getout':
            action = action_map_coinjump(action, self.args, self.prednames)
        elif self.args.m == 'bigfish':
            action = action_map_bigfish(action, self.args, self.prednames)
        elif self.args.m == 'heist':
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

        old_neural_states = torch.squeeze(torch.stack(self.buffer.neural_states, dim=0)).detach().to(device)
        old_logic_states = torch.squeeze(torch.stack(self.buffer.logic_states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_neural_states, old_logic_states,
                                                                        old_actions)

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
        if self.args.m == 'getout':
            action, explaining = self.coinjump_actor(state)
        elif self.args.m == 'bigfish':
            action, explaining = self.bigfish_actor(state)
        elif self.args.m == 'heist':
            action, explaining = self.heist_actor(state)
        return action, explaining

    def get_probs(self):
        probs = self.model.get_probs()
        return probs

    def get_explaining(self):
        explaining = 0
        return explaining

    def get_state(self, state):
        if self.args.m == 'getout':
            logic_state = extract_logic_state_coinjump(state, self.args).squeeze(0)
        elif self.args.m == 'bigfish':
            logic_state = extract_logic_state_bigfish(state, self.args).squeeze(0)
        if self.args.m == 'heist':
            logic_state = extract_logic_state_heist(state, self.args).squeeze(0)
        logic_state = logic_state.tolist()
        result = []
        for list in logic_state:
            obj_state = [round(num, 2) for num in list]
            result.append(obj_state)
        return result

    def coinjump_actor(self, coinjump):
        extracted_state = extract_logic_state_coinjump(coinjump, self.args)
        predictions = self.model(extracted_state)
        prediction = torch.argmax(predictions).cpu().item()
        explaining = self.prednames[prediction]
        action = preds_to_action_coinjump(prediction, self.prednames)
        return action, explaining

    def bigfish_actor(self, state):
        state = extract_logic_state_bigfish(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_bigfish(action, self.prednames)
        return action, explaining

    def heist_actor(self, state):
        state = extract_logic_state_heist(state, self.args)
        predictions = self.model(state)
        action = torch.argmax(predictions)
        explaining = self.prednames[action.item()]
        action = preds_to_action_heist(action, self.prednames)
        return action, explaining


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.neural_states = []
        self.logic_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.predictions = []

    def clear(self):
        del self.actions[:]
        del self.neural_states[:]
        del self.logic_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.predictions[:]

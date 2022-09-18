import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import os

import utils.replay_buffer
import utils.ppo_update

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPAgent(nn.Module):
    def __init__(self, envs):
        super(MLPAgent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 32)),
            nn.Tanh(),
            # layer_init(nn.Linear(32, 32)),
            # nn.ReLU(),
            # layer_init(nn.Linear(64, 128)),
            # nn.ReLU(),
            # layer_init(nn.Linear(128, 128)),
            # nn.ReLU(),
            # layer_init(nn.Linear(128, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh()
        )
        self.actor = layer_init(nn.Linear(32, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(32, 1), std=1)

    def forward(self, x):
        return self.network(x) 

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

class CNNAgent(nn.Module):
    def __init__(self, envs, channels=3):
        super(CNNAgent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 4, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(8*8*32, 512)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

class DDT(nn.Module):
    def __init__(self, input_dim, weights, comparators, leaves, output_dim=None, alpha=1.0, is_value=False, use_gpu=True):
        super(DDT, self).__init__()
        self.use_gpu = use_gpu
        self.leaf_init_information = leaves

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = None
        self.comparators = None
        self.selector = None

        self.init_comparators(comparators)
        self.init_weights(weights)
        self.init_alpha(alpha)
        self.init_paths()
        self.init_leaves()
        self.added_levels = nn.Sequential()

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.is_value = is_value

    def init_comparators(self, comparators):
        if comparators is None:
            comparators = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    comparators.append(np.array([1.0/self.input_dim]))
        new_comps = torch.Tensor(comparators)
        new_comps.requires_grad = True
        if self.use_gpu:
            new_comps = new_comps.cuda()
        self.comparators = nn.Parameter(new_comps)

    def init_weights(self, weights):
        if weights is None:
            weights = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4
            for level in range(depth):
                for node in range(2**level):
                    weights.append(np.random.rand(self.input_dim))

        new_weights = torch.Tensor(weights)
        new_weights.requires_grad = True
        if self.use_gpu:
            new_weights = new_weights.cuda()
        self.layers = nn.Parameter(new_weights)

    def init_alpha(self, alpha):
        self.alpha = torch.Tensor([alpha])
        if self.use_gpu:
            self.alpha = self.alpha.cuda()
        self.alpha.requires_grad = True
        self.alpha = nn.Parameter(self.alpha)

    def init_paths(self):
        if type(self.leaf_init_information) is list:
            left_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            right_branches = torch.zeros((len(self.layers), len(self.leaf_init_information)))
            for n in range(0, len(self.leaf_init_information)):
                for i in self.leaf_init_information[n][0]:
                    left_branches[i][n] = 1.0
                for j in self.leaf_init_information[n][1]:
                    right_branches[j][n] = 1.0
        else:
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            elif self.leaf_init_information is None:
                depth = 4
            left_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            for n in range(0, depth):
                row = 2 ** n - 1
                for i in range(0, 2 ** depth):
                    col = 2 ** (depth - n) * i
                    end_col = col + 2 ** (depth - 1 - n)
                    if row + i >= len(left_branches) or end_col >= len(left_branches[row]):
                        break
                    left_branches[row + i, col:end_col] = 1.0
            right_branches = torch.zeros((2 ** depth - 1, 2 ** depth))
            left_turns = np.where(left_branches == 1)
            for row in np.unique(left_turns[0]):
                cols = left_turns[1][left_turns[0] == row]
                start_pos = cols[-1] + 1
                end_pos = start_pos + len(cols)
                right_branches[row, start_pos:end_pos] = 1.0
        left_branches.requires_grad = False
        right_branches.requires_grad = False
        if self.use_gpu:
            left_branches = left_branches.cuda()
            right_branches = right_branches.cuda()
        self.left_path_sigs = left_branches
        self.right_path_sigs = right_branches

    def init_leaves(self):
        if type(self.leaf_init_information) is list:
            new_leaves = [leaf[-1] for leaf in self.leaf_init_information]
        else:
            new_leaves = []
            if type(self.leaf_init_information) is int:
                depth = int(np.floor(np.log2(self.leaf_init_information)))
            else:
                depth = 4

            last_level = np.arange(2**(depth-1)-1, 2**depth-1)
            going_left = True
            leaf_index = 0
            self.leaf_init_information = []
            for level in range(2**depth):
                curr_node = last_level[leaf_index]
                turn_left = going_left
                left_path = []
                right_path = []
                while curr_node >= 0:
                    if turn_left:
                        left_path.append(int(curr_node))
                    else:
                        right_path.append(int(curr_node))
                    prev_node = np.ceil(curr_node / 2) - 1
                    if curr_node // 2 > prev_node:
                        turn_left = False
                    else:
                        turn_left = True
                    curr_node = prev_node
                if going_left:
                    going_left = False
                else:
                    going_left = True
                    leaf_index += 1
                new_probs = np.random.uniform(0, 1, self.output_dim)  # *(1.0/self.output_dim)
                self.leaf_init_information.append([sorted(left_path), sorted(right_path), new_probs])
                new_leaves.append(new_probs)

        labels = torch.Tensor(new_leaves)
        if self.use_gpu:
            labels = labels.cuda()
        labels.requires_grad = True
        self.action_probs = nn.Parameter(labels)

    def forward(self, input_data, embedding_list=None):

        input_data = input_data.t().expand(self.layers.size(0), *input_data.t().size())

        input_data = input_data.permute(2, 0, 1)
        comp = self.layers.mul(input_data)
        comp = comp.sum(dim=2).unsqueeze(-1)
        comp = comp.sub(self.comparators.expand(input_data.size(0), *self.comparators.size()))
        comp = comp.mul(self.alpha)
        sig_vals = self.sig(comp)

        sig_vals = sig_vals.view(input_data.size(0), -1)

        one_minus_sig = torch.ones(sig_vals.size())
        if self.use_gpu:
            one_minus_sig = one_minus_sig.to('cuda')

        one_minus_sig = torch.sub(one_minus_sig, sig_vals)

        left_path_probs = self.left_path_sigs.t()
        right_path_probs = self.right_path_sigs.t()
        left_path_probs = left_path_probs.expand(input_data.size(0), *left_path_probs.size()) * sig_vals.unsqueeze(1)
        right_path_probs = right_path_probs.expand(input_data.size(0), *right_path_probs.size()) * one_minus_sig.unsqueeze(1)
        left_path_probs = left_path_probs.permute(0, 2, 1)
        right_path_probs = right_path_probs.permute(0, 2, 1)

        # We don't want 0s to ruin leaf probabilities, so replace them with 1s so they don't affect the product
        left_filler = torch.zeros(self.left_path_sigs.size())
        left_filler[self.left_path_sigs == 0] = 1
        right_filler = torch.zeros(self.right_path_sigs.size())
        if self.use_gpu:
            left_filler = left_filler.cuda()
            right_filler = right_filler.cuda()
        right_filler[self.right_path_sigs == 0] = 1

        left_path_probs = left_path_probs.add(left_filler)
        right_path_probs = right_path_probs.add(right_filler)

        probs = torch.cat((left_path_probs, right_path_probs), dim=1)
        probs = probs.prod(dim=1)
        actions = probs.mm(self.action_probs)

        if not self.is_value:
            return self.softmax(actions)
        else:
            return actions

def save_ddt(fn, model):
    checkpoint = dict()
    mdl_data = dict()
    mdl_data['weights'] = model.layers
    mdl_data['comparators'] = model.comparators
    mdl_data['leaf_init_information'] = model.leaf_init_information
    mdl_data['action_probs'] = model.action_probs
    mdl_data['alpha'] = model.alpha
    mdl_data['input_dim'] = model.input_dim
    mdl_data['is_value'] = model.is_value
    checkpoint['model_data'] = mdl_data
    torch.save(checkpoint, fn)

def load_ddt(fn):
    model_checkpoint = torch.load(fn, map_location='cpu')
    model_data = model_checkpoint['model_data']
    init_weights = [weight.detach().clone().data.cpu().numpy() for weight in model_data['weights']]
    init_comparators = [comp.detach().clone().data.cpu().numpy() for comp in model_data['comparators']]

    new_model = DDT(input_dim=model_data['input_dim'],
                    weights=init_weights,
                    comparators=init_comparators,
                    leaves=model_data['leaf_init_information'],
                    alpha=model_data['alpha'].item(),
                    is_value=model_data['is_value'])
    new_model.action_probs = model_data['action_probs']
    return new_model

def init_rule_list(num_rules, dim_in, dim_out):
    weights = np.random.rand(num_rules, dim_in)
    leaves = []
    comparators = np.random.rand(num_rules, 1)
    for leaf_index in range(num_rules):
        leaves.append([[leaf_index], np.arange(0, leaf_index).tolist(), np.random.rand(dim_out)])
    leaves.append([[], np.arange(0, num_rules).tolist(), np.random.rand(dim_out)])
    return weights, comparators, leaves

class DDTAgent:
    def __init__(self, bot_name='DDT', input_dim=4, output_dim=2, rule_list=False, num_rules=4):
        self.replay_buffer = utils.replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = bot_name
        self.rule_list = rule_list
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_rules = num_rules

        if rule_list:
            self.bot_name += str(num_rules)+'_rules'
            init_weights, init_comparators, init_leaves = init_rule_list(num_rules, input_dim, output_dim)
        else:
            init_weights = None
            init_comparators = None
            init_leaves = num_rules
            self.bot_name += str(num_rules) + '_leaves'

        self.action_network = DDT(input_dim=input_dim,
                                  output_dim=output_dim,
                                  weights=init_weights,
                                  comparators=init_comparators,
                                  leaves=init_leaves,
                                  alpha=1,
                                  is_value=False,
                                  use_gpu=False)
        self.value_network = DDT(input_dim=input_dim,
                                 output_dim=output_dim,
                                 weights=init_weights,
                                 comparators=init_comparators,
                                 leaves=init_leaves,
                                 alpha=1,
                                 is_value=True,
                                 use_gpu=False)

        self.ppo = utils.ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = None
        self.last_deep_value_pred = [None]*output_dim
        self.full_probs = None
        self.deeper_full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            # print(obs)
            # print(self.action_network.parameters)

            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs

            if self.action_network.input_dim > 10:
                probs, inds = torch.topk(probs, 3)

            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > 10:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()

        if self.action_network.input_dim > 10:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  deeper_action_log_probs=self.last_deep_action_probs,
                                  deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  deeper_full_probs_vector=self.deeper_full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, reward):
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        self.num_steps += 1
        bot_name = 'txts/' + self.bot_name
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(reward) + '\n')

    def reset(self):
        self.replay_buffer.clear()

    def save(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        save_ddt(act_fn, self.action_network)
        save_ddt(val_fn, self.value_network)

    def load(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        if os.path.exists(act_fn):
            self.action_network = load_ddt(act_fn)
            self.value_network = load_ddt(val_fn)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'bot_name': self.bot_name,
            'rule_list': self.rule_list,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'num_rules': self.num_rules
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def duplicate(self):
        new_agent = DDTAgent(bot_name=self.bot_name,
                             input_dim=self.input_dim,
                             output_dim=self.output_dim,
                             rule_list=self.rule_list,
                             num_rules=self.num_rules
                             )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
        
class DDTAgentNew:
    def __init__(self, bot_name='DDT', input_dim=4, output_dim=2, rule_list=False, num_rules=4):
        self.bot_name = bot_name
        self.rule_list = rule_list
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_rules = num_rules

        if rule_list:
            self.bot_name += str(num_rules)+'_rules'
            init_weights, init_comparators, init_leaves = init_rule_list(num_rules, input_dim, output_dim)
        else:
            init_weights = None
            init_comparators = None
            init_leaves = num_rules
            self.bot_name += str(num_rules) + '_leaves'

        self.action_network = DDT(input_dim=input_dim,
                                  output_dim=output_dim,
                                  weights=init_weights,
                                  comparators=init_comparators,
                                  leaves=init_leaves,
                                  alpha=1,
                                  is_value=False,
                                  use_gpu=False)
        self.value_network = DDT(input_dim=input_dim,
                                 output_dim=output_dim,
                                 weights=init_weights,
                                 comparators=init_comparators,
                                 leaves=init_leaves,
                                 alpha=1,
                                 is_value=True,
                                 use_gpu=False)

        self.ppo = utils.ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)

    def forward(self):
        raise NotImplementedError
    
    def get_action(self, observation, action=None):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            # print(obs)
            # print(self.action_network.parameters)

            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs

            if self.action_network.input_dim > 10:
                probs, inds = torch.topk(probs, 3)

            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > 10:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()

        if self.action_network.input_dim > 10:
            action = inds[action].item()
        else:
            action = action.item()
        return action


        # logits = self.actor(self.forward(observation))
        # probs = Categorical(logits=logits)
        # if action is None:
        #     action = probs.sample()
        # return action, probs.log_prob(action), probs.entropy()


    def get_value(self, observation):
        activation = nn.ReLU()
        last_ = nn.Linear(15, 1)
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)

            y = self.value_network(obs)
            value = last_(activation(y))
        return value

    
import torch
import torch.nn as nn
from src.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS


class LPController(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=CJA_NUM_EXPLICIT_ACTIONS, as_dict=False, special=False,
                 critic=False):
        super().__init__()
        self.as_dict = as_dict
        modules = [
            NSFRLayer()
        ]
        self.special = special

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))
        if critic:
            modules.append(torch.nn.Linear(6, out_size))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        # if not hasattr(self, "as_dict") or self.as_dict:
        #     features = torch.cat([state['base'], state['entities']], dim=1)
        # else:
        features = state
        y = self.mlp(features)
        """
        if self.special:
            encoding_base_features = 6
            Pl = encoding_base_features + 0
            #Fl = encoding_base_features + 9
            PU = encoding_base_features + 18
            En = encoding_base_features + 27
            #C0 = encoding_base_features + 36
            #C1 = encoding_base_features + 45
            st = features
            #num_players = torch.sum(st[:, Pl] != 0)
            #num_flags = torch.sum(st[:, Fl] != 0)
            #num_PUs = torch.sum(st[:, PU] != 0)
            #num_Ens = torch.sum(st[:, En] != 0)
            #num_C0 = torch.sum(st[:, C0] != 0)
            #num_C1 = torch.sum(st[:, C1] != 0)
            #print("|>", num_players, num_flags, num_PUs, num_Ens, num_C0, num_C1)
            #for i in range(20):
            #    print(st[i*50, encoding_base_features:])

            has_powerup = st[:, PU].ne(0)
            player_poweredup = st[:, Pl + 5].ne(0)
            has_enemy = st[:, En].ne(0)

            # if enemy and powerup present: reward distance to powerup
            r = (has_powerup * has_enemy) * (10 - torch.sqrt((st[:, Pl + 1] - st[:, PU + 1]) ** 2 + (st[:, Pl + 2] - st[:, PU + 2]) ** 2)/4.19)
            # if enemy and player powerupped: reward distance to enemy
            r += (player_poweredup * has_enemy) * (10 - torch.sqrt((st[:, Pl + 1] - st[:, En + 1]) ** 2 + (st[:, Pl + 2] - st[:, En + 2]) ** 2)/4.19)
            y += r.unsqueeze(-1)
        """
        return y


class NSFRLayer(nn.Module):
    def __init__(self):
        super(NSFRLayer, self).__init__()
        self.params = self.initialize_parameter()

    def forward(self, x):
        x = torch.mm(x, self.params[0])
        return x

    def initialize_parameter(self):
        parameters = torch.zeros((6, 6))
        for i in range(6):
            parameters[i][i] = torch.rand(1)
        return nn.ParameterList([nn.Parameter(parameters)])

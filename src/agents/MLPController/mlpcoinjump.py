import torch
import numpy as np


class MLPCoinjump(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=3, as_dict=False, logic=False):
        super().__init__()
        self.logic = logic
        self.as_dict = as_dict
        self.device = torch.device('cuda:0')
        encoding_base_features = 6
        encoding_entity_features = 9
        encoding_max_entities = 6
        if logic:
            self.num_in_features = 24
        else:
            self.num_in_features = encoding_base_features + encoding_entity_features * encoding_max_entities  # 60

        modules = [
            torch.nn.Linear(self.num_in_features, 40),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(40, out_size)
        ]

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        if self.logic:
            state = self.convert_states(state)

        features = state
        y = self.mlp(features)
        return y

    def convert_states(self, states):
        # TODO improve the code for better efficiency?
        states = states.cpu().numpy()
        # converted_states = torch.empty(0,device=self.device)
        converted_states = np.empty((states.shape[0], 24))
        for i, state in enumerate(states):
            temp = np.array([])
            # temp = torch.empty(0,device=self.device)
            for s in state:
                temp = np.concatenate((temp, s), axis=0)
            converted_states[i] = temp

        return torch.tensor(converted_states, dtype=torch.float32, device=self.device)

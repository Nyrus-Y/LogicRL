import torch
import numpy as np
from src.utils_bf import extract_state


class MLPController(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=5, as_dict=False, special=False):
        super().__init__()
        self.as_dict = as_dict
        self.device = torch.device('cuda:0')
        encoding_max_entities = 3
        encoding_entity_features = 3
        self.num_in_features = encoding_entity_features * encoding_max_entities  # 9

        modules = [
            torch.nn.Linear(self.num_in_features, 10),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(10, out_size)
        ]

        self.special = special

        if has_softmax:
            modules.append(torch.nn.Softmax(dim=-1))

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        state = self.convert_states(state)
        features = state
        y = self.mlp(features)
        return y

    def convert_states(self, states):
        states = states[:, :, -3:].cpu().numpy()
        # converted_states = torch.empty(0,device=self.device)
        converted_states = np.empty((states.shape[0], 9))
        for i, state in enumerate(states):
            temp = np.array([])
            # temp = torch.empty(0,device=self.device)
            for s in state:
                temp = np.concatenate((temp, s), axis=0)
            converted_states[i] = temp

        return torch.tensor(converted_states, dtype=torch.float32, device=self.device)

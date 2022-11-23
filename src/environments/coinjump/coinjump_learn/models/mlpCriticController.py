import torch
import numpy as np
from src.coinjump.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS


class MLPCriticController(torch.nn.Module):

    def __init__(self, out_size=CJA_NUM_EXPLICIT_ACTIONS):
        super().__init__()
        encoding_entity_features = 6
        encoding_max_entities = 4
        self.num_in_features = encoding_entity_features * encoding_max_entities  # 24
        self.device = "cuda:0"
        modules = [
            torch.nn.Linear(self.num_in_features, 40),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(40, out_size)
        ]

        self.mlp = torch.nn.Sequential(*modules)

    def forward(self, state):
        state = self.convert_states(state)
        y = self.mlp(state)
        return y

    def convert_states(self, states):
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

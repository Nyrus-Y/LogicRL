import torch

from src.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS


class MLPCriticController(torch.nn.Module):

    def __init__(self, out_size=CJA_NUM_EXPLICIT_ACTIONS):
        super().__init__()
        encoding_entity_features = 6
        encoding_max_entities = 4
        self.num_in_features = encoding_entity_features * encoding_max_entities  # 24

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
        converted_states = []
        for state in states:
            converted_states.append(state)
        return states

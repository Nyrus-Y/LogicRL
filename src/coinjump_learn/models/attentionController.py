import torch

from src.coinjump.coinjump.actions import CJA_NUM_EXPLICIT_ACTIONS


class AttentionController(torch.nn.Module):

    def __init__(self):
        super().__init__()

        encoding_base_features = 6
        encoding_entity_features = 9
        encoding_max_entities = 6
        self.num_in_features = encoding_base_features + encoding_entity_features * encoding_max_entities #60
        #

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_in_features, 20),
            torch.nn.Linear(20, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(5, CJA_NUM_EXPLICIT_ACTIONS)
        )

    def forward(self, x):
        state = x['state']

        features = torch.cat([state['base'], state['entities']], dim=2)
        return self.mlp(features)

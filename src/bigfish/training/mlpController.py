import torch
from src.utils_bf import extract_state


class MLPController(torch.nn.Module):

    def __init__(self, has_softmax=False, out_size=5, as_dict=False, special=False):
        super().__init__()
        self.as_dict = as_dict

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
        features = state
        y = self.mlp(features)
        return y

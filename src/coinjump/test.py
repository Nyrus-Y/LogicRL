import torch
import torch.nn as nn
import numpy as np

z = torch.tensor([[[1, 0, 0, 0, 10, 2],
                   [0, 1, 0, 0, 10, 2],
                   [0, 0, 1, 0, 10, 2],
                   [0, 0, 0, 1, 10, 2]],
                  [[1, 0, 0, 0, 10, 2],
                   [0, 1, 0, 0, 10, 2],
                   [0, 0, 1, 0, 10, 2],
                   [0, 0, 0, 1, 10, 2]],
                  [[1, 0, 0, 0, 10, 2],
                   [0, 1, 0, 0, 10, 2],
                   [0, 0, 1, 0, 10, 2],
                   [0, 0, 0, 1, 10, 2]],
                  [[1, 0, 0, 0, 10, 2],
                   [0, 1, 0, 0, 10, 2],
                   [0, 0, 1, 0, 10, 2],
                   [0, 0, 0, 1, 10, 2]],
                  [[1, 0, 0, 0, 10, 2],
                   [0, 1, 0, 0, 10, 2],
                   [0, 0, 1, 0, 10, 2],
                   [0, 0, 0, 1, 10, 2]]
                  ])
has_key = []
for i, y in enumerate(z):
    a = torch.sum(y[:, 1])
    has_key.append(a)


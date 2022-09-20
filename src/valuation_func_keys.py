import torch
import torch.nn as nn
from src.neural_utils import MLP, LogisticRegression


################################
# Valuation functions for coinjump_keys #
################################


class TypeValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, enemy）：0.87
    """

    def __init__(self):
        super(TypeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 0:4]  # [0, 1, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
        return (a * z_type).sum(dim=1)


class ClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(ClosebyValuationFunction, self).__init__()
        self.device = device

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, key2, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4:]
        c_2 = z_2[:, 4:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        dis_y = abs(c_1[:, 1] - c_2[:, 1])

        result = []
        for x, y in zip(dis_x, dis_y):
            if x < 2 and y <= 0.1:
                result.append(0.9)
            else:
                result.append(0.01)

        return torch.tensor(result)


class OnLeftValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnLeftValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4]
        c_2 = z_2[:, 4]
        diff = c_2 - c_1
        result = torch.where(diff > 0, 0.9, 0.01)
        return result
        # if c_2 - c_1 > 0:
        #     on_left = torch.tensor(0.9)
        # else:
        #     on_left = torch.tensor(0.01)
        #
        # return on_left


class OnRightValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnRightValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args: x
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4]
        c_2 = z_2[:, 4]
        diff = c_2 - c_1
        result = torch.where(diff < 0, 0.9, 0.01)
        return result
        # if c_2 - c_1 < 0:
        #     on_right = torch.tensor(0.9)
        # else:
        #     on_right = torch.tensor(0.01)
        #
        # return on_right


class HaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
               x: [agent, key, door, enemy, x, y]
               y: [agent, key1,door, enemy, key2]

        Returns:
            A batch of probabilities.
        """
        has_key = []
        for i, y in enumerate(z):
            # a = abs(1 - torch.sum(y[:, 1]))
            # has_key.append(a)
            key1 = abs(1 - y[1][1])
            key2 = abs(1 - y[4][1])
            if key1 and key2:
                a = 1
            else:
                a = 0
            has_key.append(a)
        # has_key = abs(1 - torch.sum(z[:, :, 1]))
        result = torch.tensor(has_key)

        return result


class NotHaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(NotHaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        not_has_key = []
        for i, y in enumerate(z):
            # a = torch.sum(y[:, 1]) * 0.9
            # not_has_key.append(a)

            key1 = abs(1 - y[1][1])
            key2 = abs(1 - y[4][1])
            if not key1 or not key2:
                a = 1
            else:
                a = 0
            not_has_key.append(a)
        result = torch.tensor(not_has_key)
        return result


class SafeValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(SafeValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4:]
        c_2 = z_2[:, 4:]

        # if abs(c_1[:, 1] - c_2[:, 1]) <=0.1:
        #     return torch.tensor(0)
        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        result = torch.where(dis_x > 2, 0.9, 0.01)
        return result
        # if abs(c_1[:, 0] - c_2[:, 0]) > 2:
        #     return torch.tensor(0.9)
        # else:
        #     return torch.tensor(0.01)

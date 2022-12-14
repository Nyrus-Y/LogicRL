import torch
import torch.nn as nn

################################
# Valuation functions for coinjump #
################################

device = torch.device('cuda:0')


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
        z_type = z[:, 0:4]  # [1, 0, 0, 0] * [1.0, 0, 0, 0] .sum = 0.0  type(obj1, key):0.0
        prob = (a * z_type).sum(dim=1)
        return prob


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
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4:]
        c_2 = z_2[:, 4:]

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        dis_y = abs(c_1[:, 1] - c_2[:, 1])

        result = torch.where((dis_x < 3) & (dis_y <= 0.1), 0.99, 0.1)

        return result


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
        result = torch.where(diff > 0, 0.99, 0.01)
        return result


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
        result = torch.where(diff < 0, 0.99, 0.01)
        return result


class HaveKeyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HaveKeyValuationFunction, self).__init__()

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        has_key = torch.ones(z.size(dim=0)).to(device)
        c = torch.sum(z[:, :, 1], dim=1)
        result = has_key[:] - c[:]

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
        c = torch.sum(z[:, :, 1], dim=1)
        result = c[:]

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

        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        result = torch.where(dis_x > 2, 0.99, 0.01)
        return result

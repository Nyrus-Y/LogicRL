import os

import torch
import torch.nn as nn
from src import utils_bf
from src.neural_utils import MLP, LogisticRegression

################################
# Valuation functions for bigfish #
################################


class TypeValuationFunction(nn.Module):
    """The function v_object-type
    type(obj1, agent):0.98
    type(obj2, fish）：0.87
    """

    def __init__(self):
        super(TypeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent, fish, radius, x, y]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_type = z[:, 0:2]  # [1.0, 0] * [1.0, 0] .sum = 0.0  type(obj1, agent): 1.0
        prob = (a * z_type).sum(dim=1)

        return prob


class OnTopValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnTopValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
        return result


class HighLevelValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(HighLevelValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]
        diff = c_2[:, 1] - c_1[:, 1]
        # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
        result = torch.where(diff <= 0, 99, 0)
        return result


class LowLevelValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(LowLevelValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]
        diff = c_2[:, 1] - c_1[:, 1]
        # result = utils_bf.fuzzy_position(c_2, c_1, keyword='top')
        result = torch.where(diff > 0, 99, 0)
        return result


# class OnTopLeftValuationFunction(nn.Module):
#     """The function v_closeby.
#     """
#
#     def __init__(self):
#         super(OnTopLeftValuationFunction, self).__init__()
#
#     def forward(self, z_1, z_2):
#         """
#         Args:
#             z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
#              [sfish, agent, bigfish, x, y]
#
#         Returns:
#             A batch of probabilities.
#         """
#         c_1 = z_1[:, -2:]
#         c_2 = z_2[:, -2:]
#
#         result = fuzzy_position(c_1, c_2, keyword='top_left')
#         return result


class OnLeftValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnLeftValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = utils_bf.fuzzy_position(c_2, c_1, keyword='left')
        return result


#
# class AtBottomLeftValuationFunction(nn.Module):
#     """The function v_closeby.
#     """
#
#     def __init__(self):
#         super(AtBottomLeftValuationFunction, self).__init__()
#
#     def forward(self, z_1, z_2):
#         """
#         Args:
#             z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
#              [sfish, agent, bigfish, x, y]
#
#         Returns:
#             A batch of probabilities.
#         """
#         c_1 = z_1[:, -2:]
#         c_2 = z_2[:, -2:]
#
#         result = fuzzy_position(c_1, c_2, keyword='bottom_left')
#         return result


class AtBottomValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(AtBottomValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = utils_bf.fuzzy_position(c_2, c_1, keyword='bottom')
        return result


#
# class AtBottomRightValuationFunction(nn.Module):
#     """The function v_closeby.
#     """
#
#     def __init__(self):
#         super(AtBottomRightValuationFunction, self).__init__()
#
#     def forward(self, z_1, z_2):
#         """
#         Args:
#             z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
#              [sfish, agent, bigfish, x, y]
#
#         Returns:
#             A batch of probabilities.
#         """
#         c_1 = z_1[:, -2:]
#         c_2 = z_2[:, -2:]
#
#         result = fuzzy_position(c_1, c_2, keyword='bottom_right')
#         return result


class OnRightValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(OnRightValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [sfish, agent, bigfish, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        result = utils_bf.fuzzy_position(c_2, c_1, keyword='right')
        return result


#
# class OnTopRightValuationFunction(nn.Module):
#     """The function v_closeby.
#     """
#
#     def __init__(self):
#         super(OnTopRightValuationFunction, self).__init__()
#
#     def forward(self, z_1, z_2):
#         """
#         Args:
#             z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
#              [sfish, agent, bigfish, x, y]
#
#         Returns:
#             A batch of probabilities.
#         """
#         c_1 = z_1[:, -2:]
#         c_2 = z_2[:, -2:]
#
#         result = fuzzy_position(c_1, c_2, keyword='top_right')
#         return result

class ClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    # def __init__(self, device):
    #     super(ClosebyValuationFunction, self).__init__()
    #     self.device = device
    #     self.logi = LogisticRegression(input_dim=1)
    #     # self.logi = self.initialization()
    #     self.logi.to(device)
    #
    # def forward(self, z_1, z_2):
    #     """
    #     Args:
    #         z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
    #             [agent, fish, radius, x, y]
    #         z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
    #             [agent, fish, radius, x, y]
    #     Returns:
    #         A batch of probabilities.
    #     """
    #
    #     c_1 = z_1[:, -2:]
    #     c_2 = z_2[:, -2:]
    #     r_1 = z_1[:, 2]
    #     r_2 = z_2[:, 2]
    #
    #     dis_x = torch.pow(c_2[:, 0] - c_1[:, 0], 2)
    #     dis_y = torch.pow(c_2[:, 1] - c_1[:, 1], 2)
    #     dist = torch.sqrt(dis_x[:] + dis_y[:])
    #     dist = dist[:] - r_1[:] - r_2[:]
    #     dist = dist.unsqueeze(dim=1)
    #     # dist = torch.norm(c_1 - c_2, dim=1).unsqueeze(-1)
    #     # dist = dist[:] - r_1[:] - r_2[:]
    #     probs = self.logi(dist).squeeze()
    #     return probs

    # def initialization(self):
    #     #self.logi = LogisticRegression(input_dim=1)
    #     # self.logi = NSFR_ActorCritic()
    #     self.logi.as_dict = True
    #     directory = os.getcwd()
    #     directory = os.path.join(directory, "closeby/bigfishm", "PPO_bigfishm_55844_2679.pth")
    #     a = torch.load(directory, map_location=lambda storage, loc: storage)
    #     self.logi.linear.weight = a['actor.fc.vm.layers.5.logi.linear.weight']
    #     self.logi.linear.bias = a['actor.fc.vm.layers.5.logi.linear.bias']
    #     # self.logi.load_state_dict(torch.load(directory, map_location=lambda storage, loc: storage))
    #     return self.logi

    def __init__(self, device):
        super(ClosebyValuationFunction, self).__init__()
        self.device = device

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
             [agent, fish, radius, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, -2:]
        c_2 = z_2[:, -2:]

        r_1 = z_1[:, 2]
        r_2 = z_2[:, 2]

        dis_x = torch.pow(c_2[:, 0] - c_1[:, 0], 2)
        dis_y = torch.pow(c_2[:, 1] - c_1[:, 1], 2)
        dis = torch.sqrt(dis_x[:] + dis_y[:])
        dis = abs(dis[:] - r_1[:] - r_2[:])
        probs = torch.where(dis <= 2, 0.99, 0)
        return probs


class BiggerValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(BiggerValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent,fish, radius, x, y]

        Returns:
            A batch of probabilities.
        """
        r_1 = z_1[:, 2]
        r_2 = z_2[:, 2]
        diff = r_2[:] - r_1[:]
        bigger = torch.where(diff < 0, 0.99, 0)

        return bigger


class SmallerValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self):
        super(SmallerValuationFunction, self).__init__()

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [agent,fish, radius, x, y]

        Returns:
            A batch of probabilities.
        """
        r_1 = z_1[:, 2]
        r_2 = z_2[:, 2]
        diff = r_2[:] - r_1[:]
        bigger = torch.where(diff >= 0, 0.99, 0)

        return bigger

# class NotExistBValuationFunction(nn.Module):
#     """The function v_closeby.
#     """
#
#     def __init__(self):
#         super(NotExistBValuationFunction, self).__init__()
#
#     def forward(self, z):
#         """
#         Args:
#             z (tensor): 2-d tensor B * d of object-centric representation.
#                 [sfish, agent, bigfish, x, y]
#
#         Returns:
#             A batch of probabilities.
#         """
#         has_key = []
#         for i, y in enumerate(z):
#             a = abs(1 - torch.sum(y[:, 1])) * 0.99
#             has_key.append(a)
#         # has_key = abs(1 - torch.sum(z[:, :, 1]))
#         result = torch.tensor(has_key)
#
#         return result

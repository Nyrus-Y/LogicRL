import torch
import torch.nn as nn
from src.neural_utils import MLP, LogisticRegression


################################
# Valuation functions for coinjump #
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
             [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        c_1 = z_1[:, 4:]
        c_2 = z_2[:, 4:]

        # if abs(c_1[:, 1] - c_2[:, 1]) <=0.1:
        #     return torch.tensor(0)
        dis_x = abs(c_1[:, 0] - c_2[:, 0])
        # if len(dis_x) == 1:
        #     dis_x = torch.unsqueeze(dis_x, 0)
        dis_y = abs(c_1[:, 1] - c_2[:, 1])
        # if len(dis_y) == 1:
        #     dis_y = torch.unsqueeze(dis_y, 0)

        result = []
        for x, y in zip(dis_x, dis_y):
            if x < 2 and y <= 0.1:
                result.append(0.9)
            else:
                result.append(0.01)
        # result = torch.where((dis_x < 2 and dis_y <= 0.1), 0.9, 0.01)
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
                [agent, key, door, enemy, x, y]

        Returns:
            A batch of probabilities.
        """
        has_key = []
        for i, y in enumerate(z):
            a = abs(1 - torch.sum(y[:, 1]))
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
            a = torch.sum(y[:, 1]) * 0.9
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


################################
# Valuation functions for YOLO #
################################

class YOLOColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self):
        super(YOLOColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor B * d of object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 4:7]
        return (a * z_color).sum(dim=1)


class YOLOShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self):
        super(YOLOShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 7:10]
        # a_batch = a.repeat((z.size(0), 1))  # one-hot encoding for batch
        return (a * z_shape).sum(dim=1)


class YOLOInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self):
        super(YOLOInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            x (none): A dummy argment to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        return z[:, -1]


class YOLOClosebyValuationFunction(nn.Module):
    """The function v_closeby.
    """

    def __init__(self, device):
        super(YOLOClosebyValuationFunction, self).__init__()
        self.device = device
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z_1 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]
            z_2 (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        c_1 = self.to_center(z_1)
        c_2 = self.to_center(z_2)
        dist = torch.norm(c_1 - c_2, dim=0).unsqueeze(-1)
        return self.logi(dist).squeeze()

    def to_center(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        y = (z[:, 1] + z[:, 3]) / 2
        return torch.stack((x, y))


class YOLOOnlineValuationFunction(nn.Module):
    """The function v_online.
    """

    def __init__(self, device):
        super(YOLOOnlineValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2, z_3, z_4, z_5):
        """The function to compute the probability of the online predicate.

        The colosed form f the linear regression is computed.
        The error value is fed into the 1-d logistic regression function.

        Args:
            z_i (tensor): 2-d tensor (B * D), the object-centric representation.
                [x1, y1, x2, y2, color1, color2, color3,
                    shape1, shape2, shape3, objectness]

        Returns:
            A batch of probabilities.
        """
        X = torch.stack([self.to_center_x(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        Y = torch.stack([self.to_center_y(z)
                         for z in [z_1, z_2, z_3, z_4, z_5]], dim=1).unsqueeze(-1)
        # add bias term
        X = torch.cat([torch.ones_like(X), X], dim=2)
        X_T = torch.transpose(X, 1, 2)
        # the optimal weights from the closed form solution
        W = torch.matmul(torch.matmul(
            torch.inverse(torch.matmul(X_T, X)), X_T), Y)
        diff = torch.norm(Y - torch.sum(torch.transpose(W, 1, 2)
                                        * X, dim=2).unsqueeze(-1), dim=1)
        self.diff = diff
        return self.logi(diff).squeeze()

    def to_center_x(self, z):
        x = (z[:, 0] + z[:, 2]) / 2
        return x

    def to_center_y(self, z):
        y = (z[:, 1] + z[:, 3]) / 2
        return y


##########################################
# Valuation functions for slot attention #
##########################################


class SlotAttentionInValuationFunction(nn.Module):
    """The function v_in.
    """

    def __init__(self, device):
        super(SlotAttentionInValuationFunction, self).__init__()

    def forward(self, z, x):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            x (none): A dummy argument to represent the input constant.

        Returns:
            A batch of probabilities.
        """
        # return the objectness
        return z[:, 0]


class SlotAttentionShapeValuationFunction(nn.Module):
    """The function v_shape.
    """

    def __init__(self, device):
        super(SlotAttentionShapeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_shape = z[:, 4:7]
        return (a * z_shape).sum(dim=1)


class SlotAttentionSizeValuationFunction(nn.Module):
    """The function v_size.
    """

    def __init__(self, device):
        super(SlotAttentionSizeValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_size = z[:, 7:9]
        return (a * z_size).sum(dim=1)


class SlotAttentionMaterialValuationFunction(nn.Module):
    """The function v_material.
    """

    def __init__(self, device):
        super(SlotAttentionMaterialValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_material = z[:, 9:11]
        return (a * z_material).sum(dim=1)


class SlotAttentionColorValuationFunction(nn.Module):
    """The function v_color.
    """

    def __init__(self, device):
        super(SlotAttentionColorValuationFunction, self).__init__()

    def forward(self, z, a):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
            a (tensor): The one-hot tensor that is expanded to the batch size.

        Returns:
            A batch of probabilities.
        """
        z_color = z[:, 11:19]
        return (a * z_color).sum(dim=1)


class SlotAttentionRightSideValuationFunction(nn.Module):
    """The function v_rightside.
    """

    def __init__(self, device):
        super(SlotAttentionRightSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionLeftSideValuationFunction(nn.Module):
    """The function v_leftside.
    """

    def __init__(self, device):
        super(SlotAttentionLeftSideValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=1, output_dim=1)
        self.logi.to(device)

    def forward(self, z):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        z_x = z[:, 1].unsqueeze(-1)  # (B, )
        prob = self.logi(z_x).squeeze()  # (B, )
        objectness = z[:, 0]  # (B, )
        return prob * objectness


class SlotAttentionFrontValuationFunction(nn.Module):
    """The function v_infront.
    """

    def __init__(self, device):
        super(SlotAttentionFrontValuationFunction, self).__init__()
        self.logi = LogisticRegression(input_dim=6, output_dim=1)
        self.logi.to(device)

    def forward(self, z_1, z_2):
        """
        Args:
            z (tensor): 2-d tensor (B * D), the object-centric representation.
                obj_prob + coords + shape + size + material + color
                [objectness, x, y, z, sphere, cube, cylinder, large, small, rubber,
                    metal, cyan, blue, yellow, purple, red, green, gray, brown]
        Returns:
            A batch of probabilities.
        """
        xyz_1 = z_1[:, 1:4]
        xyz_2 = z_2[:, 1:4]
        xyzxyz = torch.cat([xyz_1, xyz_2], dim=1)
        prob = self.logi(xyzxyz).squeeze()  # (B,)
        objectness = z_1[:, 0] * z_2[:, 0]  # (B,)
        return prob * objectness

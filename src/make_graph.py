import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure


def plot_weights(weights, image_directory, time_step=0):
    """plot weights of rules for bigfish"""
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(15, 5))
    plt.ylim([0, 1])
    x_label = ['up_eat', 'left_eat', 'down_eat', 'right_eat',
               'up_dodge', 'down_dodge', 'up_re', 'down_re', 'left_re', 'right_re',
               'idle_re']
    # x_label = ['Jump', 'Left_k', 'Right_k', 'Left_d',
    #            'Right_d', 'Stay', 'Jump_d', 'Left_n', 'Right_e',
    #            'Stay_n']
    x = np.arange(len(x_label)) / 2
    # width = 0.15
    # X = x - width * 2

    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()

        # X = X + width
        # plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
        plt.bar(x, W_, width=0.25, alpha=1, label='C' + str(i))
        # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))

    plt.xticks(x, x_label, fontproperties="Microsoft YaHei", size=12)
    plt.ylabel('Weights', size=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
    plt.show()
    plt.close()


# def plot_weights_beam(weights, image_directory, time_step=0):
#     weights = torch.softmax(weights, dim=1)
#     sns.set()
#     sns.set_style('white')
#     figure.Figure(figsize=(15, 5))
#     plt.ylim([0, 1])
#     # x_label = ['Jump', 'Left_key', 'Right_key', 'Left_door',
#     #            'Right_door', 'Stay', 'Jump_door', 'Left_nothing', 'Right_enemy',
#     #            'Stay_nothing']
#     x_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10',
#                'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10',
#                'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10',
#                'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'Rk7', 'RK8', 'Rk9', 'RK10',
#                'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10', ]
#     # x_label = ['Jump', 'Left_k', 'Right_k', 'Left_d',
#     #            'Right_d', 'Stay', 'Jump_d', 'Left_n', 'Right_e',
#     #            'Stay_n']
#     x = np.arange(len(x_label))
#     width = 0.15
#     X = x - width * 3
#
#     for i, W in enumerate(weights):
#         W_ = W.detach().cpu().numpy()
#
#         X = X + width
#         plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
#         # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))
#
#     plt.xticks(x, x_label, fontproperties="Microsoft YaHei", size=12)
#     plt.ylabel('Weights', size=14)
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
#     plt.show()
#     plt.close()


def plot_weights_beam(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(5, 12))
    plt.xlim([0, 1])
    # x_label = ['Jump', 'Left_key', 'Right_key', 'Left_door',
    #            'Right_door', 'Stay', 'Jump_door', 'Left_nothing', 'Right_enemy',
    #            'Stay_nothing']
    # y_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15'
    #     , 'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10', 'LK11', 'LK12', 'LK13', 'LK14', 'LK15'
    #     , 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10', 'LD11', 'LD12', 'LD13', 'LD14', 'LD15'
    #     , 'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'RK7', 'RK8', 'RK9', 'RK10', 'RK11', 'RK12', 'RK13', 'RK14', 'RK15'
    #     , 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10', 'RD11', 'RD12', 'RD13', 'RD14', 'RD15']
    # y_label = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10'
    #     , 'LK1', 'LK2', 'LK3', 'LK4', 'LK5', 'LK6', 'LK7', 'LK8', 'LK9', 'LK10'
    #     , 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 'LD8', 'LD9', 'LD10'
    #     , 'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'RK7', 'RK8', 'RK9', 'RK10'
    #     , 'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 'RD8', 'RD9', 'RD10']
    y_label = ['J1', 'J2', 'J3'
        , 'LK1', 'LK2', 'LK3'
        , 'LD1', 'LD2', 'LD3'
        , 'RK1', 'RK2', 'RK3'
        , 'RD1', 'RD2', 'RD3']
    # y_label = ['Jump', 'Left_k', 'Left_d',
    # 'Right_k', 'Right_d']
    y = np.arange(len(y_label))
    width = 0.5
    # X = x - width * 3
    for i, W in enumerate(weights):
        W_ = W.detach().cpu().numpy()

        # X = X + width
        # plt.bar(X, W_, width=width, alpha=1, label='C' + str(i))
        plt.barh(y=y, width=W_, alpha=1, label='C' + str(i))
        # plt.bar(range(len(W_)), W_, width=0.2, alpha=1, label='C' + str(i))

    plt.yticks(y, y_label, size=8)
    plt.xlabel('Weights', size=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(image_directory + 'W_' + str(time_step) + '.png', bbox_inches='tight')
    # plt.show()
    plt.close()

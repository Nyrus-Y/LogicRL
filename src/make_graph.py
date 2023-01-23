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



def plot_weights_beam(weights, image_directory, time_step=0):
    weights = torch.softmax(weights, dim=1)
    sns.set()
    sns.set_style('white')
    plt.figure(figsize=(5, 12))
    plt.xlim([0, 1])
    y_label = list(np.arange(1, 19))
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

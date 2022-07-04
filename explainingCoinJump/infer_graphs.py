import time, os, sys
import numpy as np
from string import ascii_uppercase
import pickle
import pandas as pd
import itertools
import tqdm
import random
import matplotlib.pyplot as plt

"""
Import Graph Induction Methods
"""

from methods.notears.linear import notears_linear
from methods.notears import utils
seed = 0
np.random.seed(seed)
random.seed(seed)

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc

def graph_induction(D, method):

    if method == "notears":
        G_pred = notears_linear(D, lambda1 = .1, loss_type = 'l2')
    elif method == "fci":
        G_pred = fci(D, verbose=False)
        G_pred = G_pred[0].graph
    elif method == "pc":
        G_pred = pc(D, verbose=False)
        G_pred = G_pred.G.graph

    return G_pred


"""
Loading Data
"""

data_path = "drive-download-20220502T164537Z-001/collect_coins_compact_250.pkl" # contains 250 runs of "coin collecting" behavior (human)
data = pickle.load(open(data_path, "rb"))

commonlabel = list(pd.read_csv('drive-download-20220502T164537Z-001/commonlabel.txt', sep=",", header=None, names=["ID", "Name"])["Name"])[1:]

if not os.path.exists("results_infer_graph"):
    os.makedirs("results_infer_graph")


"""
Sanity Checking Graph Induction Methods on Toy Data
"""

# # toy data example
# dims = 10
# n, d, s0, graph_type, sem_type = 1000, dims, 15, 'ER', 'gauss'
# B_true = utils.simulate_dag(d, s0, graph_type)
# W_true = utils.simulate_parameter(B_true)
# X = utils.simulate_linear_sem(W_true, n, sem_type)
#
# G_pred_nt = graph_induction(X, method="notears")
# G_pred_fci = graph_induction(X, method="fci")
# G_pred_pc = graph_induction(X, method="pc")
#
# dict_visualization = {
#     "G_pred_NT": G_pred_nt,
#     "G_pred_FCI": G_pred_fci,
#     "G_pred_PC": G_pred_pc,
# }
#
# sharex = sharey = True
# commonlabel = list(ascii_uppercase)[:dims]
# experiment_description = '''
# Sanity Check across different Graph Induction Methods\n
# '''
# suptitle = f'{experiment_description}'
#
# utils.plot_all_individual(list(dict_visualization.values()),
#                     list(dict_visualization.keys()),
#                     suptitle=suptitle,
#                     alt_form=(1,3),
#                     alt_size=(13,10),
#                     sharex=sharex,
#                     sharey=sharey,
#                     commonlabel=commonlabel)
#
# dict_cyc_vis = dict_visualization
# utils.plot_digraphs_and_cycles(list(dict_cyc_vis.values()), list(dict_cyc_vis.keys()), commonlabel)

"""
Exp 1: Average over Rollouts and Compare Maximal Disagreeing Runs
"""

methods = ["pc", "fci"]
n_rollouts = len(data)
exp1_fp = f"results_infer_graph/exp1_dict_nroll_{n_rollouts}.pkl"

if not os.path.exists(exp1_fp):

    dict_exp1 = {}
    for method in methods:

        dims = len(data[0][0][0])
        for i, rollout in enumerate(data):#tqdm.tqdm(enumerate(data), desc="Rollout "):
            if i == n_rollouts:
                break
            D = np.array(data[i][0])
            D += np.random.normal(0,1e-2,D.shape)
            G_pred = graph_induction(D, method=method)
            dict_exp1.update({f'method_{method}_rollout_{i}': G_pred})
            print(f"\t\t\t Method {method} Iteration {i}")

    pickle.dump(dict_exp1, open(exp1_fp, "wb"))

else:

    dict_exp1 = pickle.load(open(exp1_fp, "rb"))

for method in methods:

    exp1_plot_adj = f"results_infer_graph/exp1_{method}_nroll_{n_rollouts}_adj.png"
    exp1_plot_graph = f"results_infer_graph/exp1_{method}_nroll_{n_rollouts}_graph.png"

    indices = [i % n_rollouts for i, v in enumerate(dict_exp1.keys()) if method in v] # to identify values of specific method
    rollouts_names = [data_path + f"_rollout_{i}" for i in indices]
    G_pred_list = [list(dict_exp1.values())[i] for i in indices]

    G_pred_stack = np.stack(G_pred_list)
    average_G_pred = np.mean(G_pred_stack, axis=0)

    G_pred_cartesian_list = list(itertools.product(*[G_pred_list, G_pred_list]))
    rollout_cartesian_list = list(itertools.product(*[rollouts_names, rollouts_names]))
    list_of_distances = [np.linalg.norm(t[0]-t[1], ord="fro") for t in G_pred_cartesian_list]
    a = list(reversed(sorted(list_of_distances)))
    plt.hist(a, bins="auto")
    plt.title(f"{method.upper()} list of distances\nMean {np.mean(a):.2f}, Std {np.std(a):.2f} [{np.mean(a)-np.std(a):.2f},{np.mean(a)+np.std(a):.2f}], Max {np.max(a):.2f}")
    plt.savefig(f"results_infer_graph/exp1_{method}_distances.png")
    index = np.argmax(list_of_distances)
    G_pred_max_diff_tuple = G_pred_cartesian_list[index]
    max_diff_tuple_names = rollout_cartesian_list[index]

    dict_visualization = {
        "Average(G_pred)": average_G_pred,
        "binary(Average(G_pred))": utils.binary_G(average_G_pred),
        f"MD[0],\nid:{max_diff_tuple_names[0].split('_rollout_')[1]}": G_pred_max_diff_tuple[0],
        f"MD[1],id:{max_diff_tuple_names[1].split('_rollout_')[1]}": G_pred_max_diff_tuple[1],
        f"abs(MD[1]-MD[0])": abs(G_pred_max_diff_tuple[1]-G_pred_max_diff_tuple[0])
    }

    #vmax = 2*np.max(abs(W_true))
    #vmin = -vmax
    sharex = sharey = True
    # if dims > len(list(ascii_uppercase)):
    #     big_ascii = ["".join(a) for a in list(itertools.product(*[list(ascii_uppercase), list(ascii_uppercase), list(ascii_uppercase)]))]
    #     commonlabel = big_ascii[:dims]
    # else:
    #     commonlabel = list(ascii_uppercase)[:dims]
    experiment_description = f'''
    Exp 1: {method.upper()} evaluated Average over {n_rollouts} Rollouts and Compare Maximal Disagreeing Runs\n
    '''
    suptitle = f'{experiment_description}'

    utils.plot_all_individual(list(dict_visualization.values()),
                        list(dict_visualization.keys()),
                        suptitle=suptitle,
                        alt_form=(1,5),
                        alt_size=(20,8),
                        #vmin=vmin,
                        #vmax=vmax,
                        sharex=sharex,
                        sharey=sharey,
                        commonlabel=commonlabel,
                        show_text=False,
                        labelfs=3,
                        savefig=exp1_plot_adj,
                        dpi=500)

    dict_cyc_vis = dict_visualization
    # utils.plot_digraphs_and_cycles(list(dict_cyc_vis.values()), list(dict_cyc_vis.keys()), commonlabel,
    #                     alt_size=(18,18), arrowsize=7, font_size=6, node_size=150, no_cycles=True, savefig=exp1_plot_graph, dpi=500)
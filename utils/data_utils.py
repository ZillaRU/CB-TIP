# import pickle
import os

import dill
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve


def serialize(data):
    data_tuple = tuple(data.values())
    return dill.dumps(data_tuple)


def deserialize_small(data):
    keys = ('mol_graph', 'graph_size')
    return dict(zip(keys, dill.loads(data)))


def deserialize_macro(data):
    keys = ('seq', 'mol_graph')
    # print(dill.loads(data))
    return dict(zip(keys, dill.loads(data)))


def gen_preds(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
    return preds, preds_neg


def eval_threshold(labels_all, preds_all):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2*i] > 0.95 and preds_all[2*i+1] > 0.95:
            preds_all[2*i] = max(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
        else:
            preds_all[2*i] = min(preds_all[2*i], preds_all[2*i+1])
            preds_all[2*i+1] = preds_all[2*i]
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def convert_triplets_to_Xy(triplets):
    X1 = np.array(triplets['pos'], dtype=np.int16)
    X0 = np.array(triplets['neg'], dtype=np.int16)
    X1 = np.concatenate((X1, np.ones(shape=(X1.shape[0], 1), dtype=np.int16)), 1)
    X0 = np.concatenate((X0, np.zeros(shape=(X0.shape[0], 1), dtype=np.int16)), 1)
    return np.concatenate((X1, X0), 0)


def dd_dt_tt_convert_triplets_to_Xy(triplets):
    Xy_dd_dt_tt = {}
    for i in ['dd', 'dt', 'tt']:
        X1 = np.array(triplets['pos'][i], dtype=np.int16)
        X0 = np.array(triplets['neg'][i], dtype=np.int16)
        X1 = np.concatenate((X1, np.ones(shape=(X1.shape[0], 1), dtype=np.int16)), 1)
        X0 = np.concatenate((X0, np.zeros(shape=(X0.shape[0], 1), dtype=np.int16)), 1)
        Xy_dd_dt_tt[i] = np.concatenate((X1, X0), 0)
    return Xy_dd_dt_tt


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

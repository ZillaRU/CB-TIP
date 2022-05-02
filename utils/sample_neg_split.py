import os

import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
from scipy.special import softmax
from tqdm import tqdm

# valid_drugs = set(pd.read_csv('full_drugbank_final_drugs.csv', header=None).iloc[:, 0])
# ddis = pd.read_csv('full_pos.csv', header=None).values.tolist()
# new_ddis = []
# for s, r, t in ddis:
#     if s in valid_drugs and t in valid_drugs:
#         new_ddis.append([s, r, t])
#     else:
#         print("???")
# pd.DataFrame(new_ddis).to_csv('all_pos_deep_ddi.csv', index=False, header=None)


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # # if max_size is set, randomly sample train links
    # if max_size < len(pos_edges):
    #     perm = np.random.permutation(len(pos_edges))[:max_size]
    #     pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, rel, neg_tail])
            pbar.update(1)

    pbar.close()

    return neg_edges


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split(',') for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


# adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files({'train': 'full_pos.csv'})
# print(len(triplets['train']))
# neg_es = sample_neg(adj_list, triplets['train'])
# with open('full_neg.csv', 'w') as f:
#     for h, r, t in neg_es:
#         h = id2entity[h]
#         r = id2relation[r]
#         t = id2entity[t]
#         f.write(f'{h},{r},{t}\n')


def random_split(path, rate):
    import random
    with open('all_pos_deep_ddi.csv', 'r') as f:
        pos = [line.split(',') for line in f.read().split('\n')[:-1]]
    random.shuffle(pos)
    pos = pd.DataFrame(pos, index=None)
    sp1, sp2 = int(rate[0] * len(pos)), int((1 - rate[2]) * len(pos))

    _path = path+'deep118-1/'
    os.mkdir(_path)

    pos.loc[:sp1].to_csv(_path + 'train_pos.txt', sep=',', index=False, header=None)
    pos.loc[sp1:sp2].to_csv(_path + 'valid_pos.txt', sep=',', index=False, header=None)
    pos.loc[sp2:].to_csv(_path + 'test_pos.txt', sep=',', index=False, header=None)

    with open('all_neg_deep_ddi.csv', 'r') as f:
        neg = [line.split(',') for line in f.read().split('\n')[:-1]]
    random.shuffle(neg)
    neg = pd.DataFrame(neg, index=None)
    sp1, sp2 = int(rate[0] * len(neg)), int((1 - rate[2]) * len(neg))

    neg.loc[:sp1].to_csv(_path + 'train_neg.txt', sep=',', index=False, header=None)
    neg.loc[sp1:sp2].to_csv(_path + 'valid_neg.txt', sep=',', index=False, header=None)
    neg.loc[sp2:].to_csv(_path + 'test_neg.txt', sep=',', index=False, header=None)


random_split('', rate=[0.1, 0.1, 0.8])
import dgl
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix


def dd_dt_tt_build_inter_graph_from_links(dataset, tt_types, dt_types, split, saved_relation2id=None):
    files = {
        'train': {
            'pos': f'data/{dataset}/{split}/train_pos.txt',
            'neg': f'data/{dataset}/{split}/train_neg.txt'
        },
        'valid': {
            'pos': f'data/{dataset}/{split}/valid_pos.txt',
            'neg': f'data/{dataset}/{split}/train_neg.txt'
        }
    }
    drug2id, target2id = {}, {}

    if dataset == 'full_drugbank':
        biodrug_set = set(pd.read_csv(f'data/{dataset}/drug_seqs.csv', header=None).iloc[:, 0])
        bio_cnt = 0
        bio2id = {}
        # dd_types
        # ud,ub,vd,vb
        type_dict = {
            (True, True, True, True): ('macro', 'macro'),
            (True, False, True, False): ('small', 'small'),
            (True, False, True, True): ('small', 'macro'),
            (True, True, True, False): ('macro', 'small'),
            (True, True, False, False): ('macro', 'target'),
            (True, False, False, False): ('small', 'target')
        }
    else:
        type_dict = {
            (True, True): 'dd',
            (True, False): 'dt',
            (False, False): 'tt'
        }
        # dd_types

    relation2id = {} if not saved_relation2id else None

    small_cnt, target_cnt = 0, 0
    rel = 0
    triplets = {}
    dd_dt_tt_triplets = {}

    for file_type, file_paths in files.items():  # train/valid/test, pos/neg
        triplets[file_type] = {}
        dd_dt_tt_triplets[file_type] = {}
        for y, path in file_paths.items():  # pos/neg, path
            if dataset == 'full_drugbank':  # asymmetric
                dd_dt_tt_data= {}
                dd_dt_tt_data[('small', 'small')] = []
                dd_dt_tt_data[('small', 'macro')] = []
                dd_dt_tt_data[('macro', 'small')] = []
                dd_dt_tt_data[('macro', 'macro')] = []
                dd_dt_tt_data[('macro', 'target')] = []
                dd_dt_tt_data[('small', 'target')] = []
            else:
                dd_dt_tt_data = {
                    'dd': [],
                    'dt': [],
                    'tt': []
                }
            data = []
            with open(path) as f:
                file_data = [line.split(',') for line in f.read().split('\n')[:-1]]
            for [u, r, v] in file_data:
                if r == 'dt':
                    u_is_d, v_is_d = True, False
                elif r == 'dd':
                    u_is_d, v_is_d = True, True
                elif r == 'tt':
                    u_is_d, v_is_d = False, False
                elif dataset == 'full_drugbank':  # for drugbank
                    u_is_d, v_is_d = u.startswith('DB'), v.startswith('DB')
                    u_is_bio, v_is_bio = u in biodrug_set, v in biodrug_set
                else:
                    raise NotImplementedError
                if dataset != 'full_drugbank':
                    if u_is_d:
                        if u not in drug2id:
                            uid = drug2id[u] = small_cnt
                            small_cnt += 1
                        else:
                            uid = drug2id[u]
                    elif u not in target2id:
                        uid = target2id[u] = target_cnt
                        target_cnt += 1
                    else:
                        uid = target2id[u]
                    if v_is_d:
                        if v not in drug2id:
                            vid = drug2id[v] = small_cnt
                            small_cnt += 1
                        else:
                            vid = drug2id[v]
                    elif v not in target2id:
                        vid = target2id[v] = target_cnt
                        target_cnt += 1
                    else:
                        vid = target2id[v]
                else:
                    if u_is_d:
                        if u_is_bio:
                            if u not in bio2id:
                                uid = bio2id[u] = bio_cnt
                                bio_cnt += 1
                            else:
                                uid = bio2id[u]
                        else:
                            if u not in drug2id:
                                uid = drug2id[u] = small_cnt
                                small_cnt += 1
                            else:
                                uid = drug2id[u]
                    elif u not in target2id:
                        uid = target2id[u] = target_cnt
                        target_cnt += 1
                    else:
                        uid = target2id[u]
                    if v_is_d:
                        if v_is_bio:
                            if v not in bio2id:
                                vid = bio2id[v] = bio_cnt
                                bio_cnt += 1
                            else:
                                vid = bio2id[v]
                        else:
                            if v not in drug2id:
                                vid = drug2id[v] = small_cnt
                                small_cnt += 1
                            else:
                                vid = drug2id[v]
                    elif v not in target2id:
                        vid = target2id[v] = target_cnt
                        target_cnt += 1
                    else:
                        vid = target2id[v]
                if not saved_relation2id and r not in relation2id:
                    relation2id[r] = rel
                    rel += 1
                # Save the triplets corresponding to only the known relations
                if r in relation2id:
                    temp = [uid, vid, relation2id[r]]
                    if dataset != 'full_drugbank':
                        data.append(temp)
                        dd_dt_tt_data[type_dict[(u_is_d, v_is_d)]].append(temp)
                    else:
                        dd_dt_tt_data[type_dict[(u_is_d, u_is_bio, v_is_d, v_is_bio)]].append(temp)
            if dataset != 'full_drugbank':
                triplets[file_type][y] = np.array(data, dtype=np.int16)
            dd_dt_tt_triplets[file_type][y] = dd_dt_tt_data
        if dataset == 'full_drugbank':
            for _set, label_tris in dd_dt_tt_triplets.items():
                for _label, dd_dt_tt_data in label_tris.items():
                    temp_list = dd_dt_tt_data[('small', 'macro')]
                    dd_dt_tt_data[('small', 'macro')] = [
                        [u, v + small_cnt, r] for u, v, r in temp_list
                    ]
                    temp_list = dd_dt_tt_data[('macro', 'small')]
                    dd_dt_tt_data[('macro', 'small')] = [
                        [u + small_cnt, v, r] for u, v, r in temp_list
                    ]
                    temp_list = dd_dt_tt_data[('macro', 'target')]
                    dd_dt_tt_data[('macro', 'target')] = [
                        [u + small_cnt, v, r] for u, v, r in temp_list
                    ]
                    temp_list = dd_dt_tt_data[('target', 'macro')]
                    dd_dt_tt_data[('target', 'macro')] = [
                        [u, v + small_cnt, r] for u, v, r in temp_list
                    ]
                    temp_list = dd_dt_tt_data[('macro', 'macro')]
                    dd_dt_tt_data[('macro', 'macro')] = [
                        [u + small_cnt, v + small_cnt, r] for u, v, r in temp_list
                    ]
                    temp_dict = {}
                    temp_dict['dd'] = dd_dt_tt_data[('small', 'small')] + dd_dt_tt_data[('small', 'macro')] \
                                      + dd_dt_tt_data[('macro', 'small')] + dd_dt_tt_data[('macro', 'macro')]
                    temp_dict['dt'] = dd_dt_tt_data[('small', 'target')] + dd_dt_tt_data[('macro', 'target')]
                    temp_dict['tt'] = dd_dt_tt_data[('target', 'target')]
                    dd_dt_tt_triplets[_set][_label] = temp_dict
                    triplets[_set][_label] = np.array(temp_dict['dd'] + temp_dict['dt'] + temp_dict['tt'],
                                                      dtype=np.int16)
            for k, v in bio2id:
                drug2id[small_cnt + k] = v
    id2drug = {v: k for k, v in drug2id.items()}
    id2target = {v: k for k, v in target2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation.
    # Note that this is constructed only from the train data.
    _dict = {}
    for _set in ['pos', 'neg']:
        _dict[_set] = {}
        for i in range(len(relation2id)):
            idx = np.argwhere(triplets['train'][_set][:, 2] == i)
            rel = id2relation[i]
            rel_tuple = (
                "target" if (rel in tt_types or rel in dt_types) else "drug",
                rel,
                "drug" if rel not in tt_types else "target"
            ) if rel != 'dt' else ('drug', 'dt', 'target')
            shape = (
                target_cnt if (rel in tt_types or rel in dt_types) else small_cnt,
                small_cnt if rel not in tt_types else target_cnt
            )
            if rel == 'dt':
                shape = (small_cnt, target_cnt)
            print(rel, shape)
            _dict[_set][rel_tuple] = csc_matrix(
                (
                    np.ones(len(idx), dtype=np.uint8),
                    (
                        triplets['train'][_set][:, 0][idx].squeeze(1),
                        triplets['train'][_set][:, 1][idx].squeeze(1)
                    )
                ), shape=shape)
        drug_cnt = small_cnt + (bio_cnt if dataset == 'full_drugbank' else 0)
        print(f'drug_cnt: {drug_cnt}, (including {small_cnt} small ones); target_cnt: {target_cnt}')
    return _dict['pos'], _dict['neg'], triplets, dd_dt_tt_triplets, \
           drug2id, target2id, relation2id, \
           id2drug, id2target, id2relation, \
           drug_cnt, target_cnt, small_cnt


def ssp_multigraph_to_dgl(adjs, relation2id):
    adjs = {k: v.tocoo() for k, v in adjs.items()}
    # g_dgl = dgl.heterograph({
    #     k: (torch.from_numpy(v.row), torch.from_numpy(v.col)) for k, v in adjs.items()
    # })
    # return dgl.to_bidirected(g_dgl)
    graph_dict = {}
    for k, v in adjs.items():
        print(k)
        if k[0] != k[2]:
            graph_dict[k] = (torch.from_numpy(v.row), torch.from_numpy(v.col))
            graph_dict[(k[2], f"~{k[1]}", k[0])] = (torch.from_numpy(v.col), torch.from_numpy(v.row))
            relation2id[f"~{k[1]}"] = len(relation2id)
        else:
            # graph_dict[k] = (torch.from_numpy(np.hstack((v.row, v.col))),
            #                  torch.from_numpy(np.hstack((v.col, v.row))))
            graph_dict[k] = (torch.from_numpy(v.row),
                             torch.from_numpy(v.col))
    # g_dgl = dgl.heterograph({
    #     k: (torch.from_numpy(np.hstack((v.row, v.col))),
    #         torch.from_numpy(np.hstack((v.col, v.row))))
    #     for k, v in adjs.items()
    # })
    g_dgl = dgl.heterograph(graph_dict)
    return g_dgl, relation2id

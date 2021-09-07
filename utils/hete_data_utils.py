import dgl
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csc_matrix


def dd_dt_tt_build_inter_graph_from_links(dataset, split, saved_relation2id=None):
    _pre = '../baselines/data/'
    files = {
        'train': {
            'pos': f'{_pre}{dataset}{split}/train_pos.txt',
            'neg': f'{_pre}{dataset}{split}/train_neg.txt'
        },
        'valid': {
            'pos': f'{_pre}{dataset}{split}/valid_pos.txt',
            'neg': f'{_pre}{dataset}{split}/train_neg.txt'
        },
        'test': {
            'pos': f'{_pre}{dataset}{split}/test_pos.txt',
            'neg': f'{_pre}{dataset}{split}/test_neg.txt'
        }
    }

    dti_file = f'data/{dataset}/DTIs.csv'
    tti_file = f'data/{dataset}/TTIs.csv'

    drug2id, bio2id, target2id = {}, {}, {}

    biodrug_set = set(pd.read_csv(f'data/{dataset}/biotech_seqs.csv', header=None).iloc[:, 0]) \
        if dataset == 'full' else set()

    # dd_types
    # ub,vd,vb
    type_dict = {
        (True, True): ('macro', 'macro'),
        (False, False): ('small', 'small'),
        (False, True): ('small', 'macro'),
        (True, False): ('macro', 'small')
    }

    relation2id = {} if not saved_relation2id else None

    small_cnt, bio_cnt, target_cnt = 0, 0, 0
    rel = 0
    triplets = {}
    dd_dt_tt_triplets = {}

    # all triplets are DDIs
    for file_type, file_paths in files.items():  # train/valid/test, pos/neg
        triplets[file_type] = {}
        dd_dt_tt_triplets[file_type] = {}
        for y, path in file_paths.items():  # pos/neg, path
            dd_dt_tt_data = {}
            dd_dt_tt_data[('small', 'small')] = []
            dd_dt_tt_data[('small', 'macro')] = []
            dd_dt_tt_data[('macro', 'small')] = []
            dd_dt_tt_data[('macro', 'macro')] = []
            with open(path) as f:
                file_data = [line.split(',') for line in f.read().split('\n')[:-1]]
            for u, r, v in file_data:
                u_is_bio, v_is_bio = u in biodrug_set, v in biodrug_set
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
                if not saved_relation2id and r not in relation2id:
                    relation2id[r] = rel
                    rel += 1
                # Save the triplets corresponding to only the known relations
                if r in relation2id:
                    temp = [uid, vid, relation2id[r]]
                    dd_dt_tt_data[type_dict[(u_is_bio, v_is_bio)]].append(temp)
            dd_dt_tt_triplets[file_type][y] = dd_dt_tt_data
        if file_type == 'train':
            dti_list = pd.read_csv(dti_file, header=None).values.tolist()
            small_t_is, bio_t_is = [], []
            for d, t in dti_list:
                d_is_bio = d in biodrug_set
                if d_is_bio:
                    if d not in bio2id:
                        did = bio2id[d] = bio_cnt
                        bio_cnt += 1
                    else:
                        did = bio2id[d]
                else:
                    if d not in drug2id:
                        did = drug2id[d] = small_cnt
                        small_cnt += 1
                    else:
                        did = drug2id[d]
                if t not in target2id:
                    tid = target2id[t] = target_cnt
                    target_cnt += 1
                else:
                    tid = target2id[t]
                if d_is_bio:
                    bio_t_is.append([did, tid, rel])
                else:
                    small_t_is.append([did, tid, rel])
            dd_dt_tt_triplets[file_type]['pos'][('small', 'target')] = small_t_is
            dd_dt_tt_triplets[file_type]['pos'][('macro', 'target')] = bio_t_is
            relation2id['dt'] = rel
            rel += 1

            tti_list = pd.read_csv(tti_file, header=None).values.tolist()
            ttis = []
            for t1, t2 in tti_list:
                if t1 not in target2id:
                    tid1 = target2id[t1] = target_cnt
                    target_cnt += 1
                else:
                    tid1 = target2id[t1]
                if t2 not in target2id:
                    tid2 = target2id[t2] = target_cnt
                    target_cnt += 1
                else:
                    tid2 = target2id[t2]
                ttis.append([tid1, tid2, rel])
            dd_dt_tt_triplets[file_type]['pos'][('target', 'target')] = ttis
            relation2id['tt'] = rel
            rel += 1

    for _set, label_tris in dd_dt_tt_triplets.items():
        # print(_set, label_tris.keys())
        for _label, dd_dt_tt_data in label_tris.items():
            # print(_set, _label, dd_dt_tt_data.keys())
            temp_list = dd_dt_tt_data[('small', 'macro')]
            dd_dt_tt_data[('small', 'macro')] = [
                [u, v + small_cnt, r] for u, v, r in temp_list
            ]
            temp_list = dd_dt_tt_data[('macro', 'small')]
            dd_dt_tt_data[('macro', 'small')] = [
                [u + small_cnt, v, r] for u, v, r in temp_list
            ]
            temp_list = dd_dt_tt_data[('macro', 'macro')]
            dd_dt_tt_data[('macro', 'macro')] = [
                [u + small_cnt, v + small_cnt, r] for u, v, r in temp_list
            ]
            temp_dict = {}
            if _set == 'train' and _label == 'pos':
                temp_list = dd_dt_tt_data[('macro', 'target')]
                dd_dt_tt_data[('macro', 'target')] = [
                    [u + small_cnt, v, r] for u, v, r in temp_list
                ]
                temp_dict['dd'] = dd_dt_tt_data[('small', 'small')] + dd_dt_tt_data[('small', 'macro')] \
                                  + dd_dt_tt_data[('macro', 'small')] + dd_dt_tt_data[('macro', 'macro')]
                temp_dict['dt'] = dd_dt_tt_data[('small', 'target')] + dd_dt_tt_data[('macro', 'target')]
                temp_dict['tt'] = dd_dt_tt_data[('target', 'target')]
                triplets[_set][_label] = np.array(temp_dict['dd'] + temp_dict['dt'] + temp_dict['tt'],
                                                  dtype=np.int16)
            else:
                temp_dict['dd'] = dd_dt_tt_data[('small', 'small')] + dd_dt_tt_data[('small', 'macro')] \
                                  + dd_dt_tt_data[('macro', 'small')] + dd_dt_tt_data[('macro', 'macro')]
                triplets[_set][_label] = np.array(temp_dict['dd'], dtype=np.int16)
            dd_dt_tt_triplets[_set][_label] = temp_dict
    for k, v in bio2id.items():
        drug2id[k] = v + small_cnt
    id2drug = {v: k for k, v in drug2id.items()}
    id2target = {v: k for k, v in target2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    dt_rel_id, tt_rel_id = 'dt', 'tt'
    drug_cnt = small_cnt + bio_cnt
    # print(drug_cnt, small_cnt, bio_cnt, target_cnt)
    # Construct the list of adjacency matrix each corresponding to each relation.
    # Note that this is constructed only from the train data.
    _dict = {}
    for _set in ['pos', 'neg']:
        _dict[_set] = {}
        for i in range(len(relation2id)):
            idx = np.argwhere(triplets['train'][_set][:, 2] == i)
            rel = id2relation[i]

            rel_tuple = (
                "target" if rel == tt_rel_id else "drug",
                rel,
                "target" if (rel == tt_rel_id or rel == dt_rel_id) else "drug"
            )
            shape = (
                drug_cnt if rel != tt_rel_id else target_cnt,
                target_cnt if (rel == tt_rel_id or rel == dt_rel_id) else drug_cnt
            )
            # print(_set, rel, shape)
            _dict[_set][rel_tuple] = csc_matrix(
                (
                    np.ones(len(idx), dtype=np.uint8),
                    (
                        triplets['train'][_set][:, 0][idx].squeeze(1),
                        triplets['train'][_set][:, 1][idx].squeeze(1)
                    )
                ), shape=shape)
    print(f'drug_cnt: {drug_cnt}, (including {small_cnt} small ones); target_cnt: {target_cnt}')
    return _dict['pos'], _dict['neg'], triplets, dd_dt_tt_triplets, \
           drug2id, target2id, relation2id, \
           id2drug, id2target, id2relation, \
           drug_cnt, target_cnt, small_cnt


def ssp_multigraph_to_dgl(drug_cnt, target_cnt, adjs, relation2id):
    adjs = {k: v.tocoo() for k, v in adjs.items()}
    # g_dgl = dgl.heterograph({
    #     k: (torch.from_numpy(v.row), torch.from_numpy(v.col)) for k, v in adjs.items()
    # })
    # return dgl.to_bidirected(g_dgl)
    graph_dict = {}
    for k, v in adjs.items():
        # print(k, v)
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
    # CAUTION: must assign the num_nodes_dict explicitly, since there are isolated molecules in pos_train_graph
    g_dgl = dgl.heterograph(graph_dict,
                            num_nodes_dict={
                                'drug': drug_cnt,
                                'target': target_cnt
                            })
    return g_dgl, relation2id


def build_valid_test_graph(drug_cnt, edges, relation2id, id2relation):
    adjs = {}

    for i in range(len(relation2id)-2):
        idx = np.argwhere(edges[:, 2] == i)
        rel = id2relation[i]
        rel_tuple = ("drug", rel, "drug")
        shape = (drug_cnt, drug_cnt)
        # print(rel, shape)
        adjs[rel_tuple] = csc_matrix(
            (
                np.ones(len(idx), dtype=np.uint8),
                (
                    edges[:, 0][idx].squeeze(1),
                    edges[:, 1][idx].squeeze(1)
                )
            ), shape=shape).tocoo()
    graph_dict = {}
    for k, v in adjs.items():
        # print(k, v)
        if k[0] != k[2]:
            graph_dict[k] = (torch.from_numpy(v.row), torch.from_numpy(v.col))
            graph_dict[(k[2], f"~{k[1]}", k[0])] = (torch.from_numpy(v.col), torch.from_numpy(v.row))
            relation2id[f"~{k[1]}"] = len(relation2id)
        else:
            # graph_dict[k] = (torch.from_numpy(np.hstack((v.row, v.col))),
            #                  torch.from_numpy(np.hstack((v.col, v.row))))
            graph_dict[k] = (torch.from_numpy(v.row),
                             torch.from_numpy(v.col))
        # gdgl = dgl.heterograph({
        #     k: (torch.from_numpy(np.hstack((v.row, v.col))),
        #         torch.from_numpy(np.hstack((v.col, v.row))))
        #     for k, v in adjs.items()
        # })
        # CAUTION: must assign the num_nodes_dict explicitly, since there are isolated molecules in pos_train_graph
    return dgl.heterograph(graph_dict,
                           num_nodes_dict={
                               'drug': drug_cnt
                           })

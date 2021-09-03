import logging
import os
import time

import pandas as pd
import torch
from sklearn import metrics

from model import build_optimizer
from model.model_config import initialize_BioMIP
from model.customized_loss import select_loss_function
from utils.arg_parser import parser
from utils.data_utils import eval_threshold
from utils.generate_intra_graph_db import generate_small_mol_graph_datasets, generate_macro_mol_graph_datasets
from utils.hete_data_utils import ssp_multigraph_to_dgl, \
    dd_dt_tt_build_inter_graph_from_links
from utils.intra_graph_dataset import IntraGraphDataset


# training fuction on each epoch
def train(encoder, decoder, dgi_model,
          opt, loss_fn: dict,
          mol_graphs,
          train_pos_graph, train_neg_graph,
          pred_rels):
    encoder.train()
    decoder.train()
    dgi_model.train()
    opt.zero_grad()
    # emb_intra: dict, keys: "small", "bio", "target"
    # emb_inter: dict, keys: "drug", "target"
    emb_intra, emb_inter = encoder(mol_graphs,
                                   train_pos_graph,
                                   pred_rels=pred_rels)

    dgi_loss = dgi_model(emb_intra, emb_inter,
                         train_pos_graph, train_neg_graph)

    # decoder:  return dict,
    #           keys: intra, inter, mix
    pos_scores = decoder(train_pos_graph, emb_intra, emb_inter,
                         pred_rels=pred_rels)
    neg_scores = decoder(train_neg_graph, emb_intra, emb_inter,
                         pred_rels=pred_rels)
    labels = torch.cat([torch.ones(pos_scores[0].shape[0]), torch.zeros(neg_scores[0].shape[0])]).numpy()
    losses = {'dgi': dgi_loss}
    for i in ('intra', 'inter', 'mix'):
        losses[i] = loss_fn[i](torch.cat([pos_scores[i], neg_scores[i]]).numpy(), labels)
    curr_loss = torch.stack(list(losses.values())).sum()
    curr_loss.backward()
    opt.step()
    print('Train epoch: {} Loss: {:.6f}'.format(epoch, curr_loss.item()))
    return emb_intra, emb_inter


# # training fuction on each epoch
# def train_both(encoder, decoder, opt, loss_fn: dict, mol_graphs, train_pos_graph, train_neg_graph, pred_rels):
#     encoder.train()
#     decoder.train()
#     opt.zero_grad()
#     emb_intra, emb_inter = encoder(mol_graphs,
#                                    train_pos_graph)
#     pos_score_intra, pos_score_inter = decoder(train_pos_graph, emb_intra, emb_inter,
#                                                pred_rels=pred_rels)
#     neg_score_intra, neg_score_inter = decoder(train_neg_graph, emb_intra, emb_inter,
#                                                pred_rels=pred_rels)
#     print(pos_score_intra)
#     print(neg_score_intra)
#     print(pos_score_inter)
#     print(neg_score_inter)
#     labels = torch.cat([torch.ones(pos_score_inter.shape[0]), torch.zeros(neg_score_inter.shape[0])]).numpy()
#     intra_scores = torch.cat([pos_score_intra, neg_score_intra]).numpy()
#     inter_scores = torch.cat([pos_score_inter, neg_score_inter]).numpy()
#
#     # encoder:
#     # - GNN for small molecules (attentive FP or ...)
#     # - CNN/RNN/GNN for macro molecules
#
#     # decoder:
#     # - Decagon(DistMulti/Bilinear/Dot) multi-relational
#
#     # multi-view alignment straightforward extension for heterogeneous graphs?
#     # - MIRACLE-DGIloss Deep Graph Infomax (mutual information maximization) https://github.com/dmlc/dgl/blob/master/examples/pytorch/dgi/dgi.py
#     # - DEAL(Dual Encoder with ALignment (mutual information maximization)
#     #   the idea of loose-alignment is similar with constrasive learning applied in MIRACLE
#
#     # handling missing intra-/inter-view): DEAL
#
#     loss = loss_fn['BCE'](labels, scores, )
#     if 'ERROR' in loss_fn:
#         loss += loss_fn['DIFF'](emb_intra)
#     # loss = loss_fn(output, data.y.view(-1, 1).float().to(device)).mean()
#     loss.backward()
#     opt.step()
#     print('Train epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
#     return emb_intra, emb_inter


def predicting(model,
               intra_feats, inter_feats,
               pos_edges, neg_edges):
    pos_score, neg_score = model(intra_feats, inter_feats, pos_edges, neg_edges)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return labels, torch.cat([pos_score, neg_score]).numpy()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    # initialize_experiment(params)
    # save featurized intra-view graphs
    print(params)

    time_str = time.strftime('%m-%d %H:%M', time.localtime(time.time()))

    if params.dataset == 'full_drugbank':
        params.aln_path = '/data/rzy/drugbank_prot/full_drugbank/aln'
        params.npy_path = '/data/rzy/drugbank_prot/full_drugbank/pconsc4'
        params.small_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'st_drugbank':
        params.aln_path = '/data/rzy/drugbank_prot/full_drugbank/aln'
        params.npy_path = '/data/rzy/drugbank_prot/full_drugbank/pconsc4'
        params.small_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'davis':
        params.aln_path = '/data/rzy/davis/aln'
        params.npy_path = '/data/rzy/davispconsc4'
        params.small_mol_db_path = f'/data/rzy/davis/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/davis/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'kiba':
        params.aln_path = '/data/rzy/kiba/aln'
        params.npy_path = '/data/rzy/kibapconsc4'
        params.small_mol_db_path = f'/data/rzy/kiba/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/kiba/prot_graph_db'  # _{params.prot_featurizer}
    else:
        raise NotImplementedError

    macro_mol_list = pd.read_csv(f'data/{params.dataset}/macro_seqs.csv', header=None, names=['id', '_'])[
        'id'].tolist()
    macro_mol_list = [str(i) for i in macro_mol_list]  # assure mol_id is a string

    print('small molecule db_path:', params.small_mol_db_path)
    print('macro molecule db_path:', params.macro_mol_db_path)

    # if not processed, build intra-view graphs
    if not os.path.isdir(params.small_mol_db_path):
        generate_small_mol_graph_datasets(params)
    if not os.path.isdir(params.macro_mol_db_path):
        generate_macro_mol_graph_datasets(params)

    # load intra-view graph dataset
    small_mol_graphs = IntraGraphDataset(db_path=params.small_mol_db_path, db_name='small_mol')
    macro_mol_graphs = IntraGraphDataset(db_path=params.macro_mol_db_path, db_name='macro_mol')

    params.atom_insize = small_mol_graphs.get_nfeat_dim()
    params.bond_insize = small_mol_graphs.get_efeat_dim()
    params.aa_node_insize = macro_mol_graphs.get_nfeat_dim()
    params.aa_edge_insize = macro_mol_graphs.get_efeat_dim()

    # edge_type2decoder = {
    #     'tt': 'dedicom',
    #     'dt': 'bilinear',
    #     '~dt': 'bilinear',
    #     'dd': 'dedicom'
    # }

    # load the inter-view graph
    pos_adj_dict, neg_adj_dict, \
    triplets, dd_dt_tt_triplets, \
    drug2id, target2id, relation2id, \
    id2drug, id2target, id2relation, \
    drug_cnt, target_cnt, small_cnt = dd_dt_tt_build_inter_graph_from_links(
        dataset=params.dataset,
        split=params.split
    )
    # init pos/neg inter-graph
    train_pos_graph, train_neg_graph = ssp_multigraph_to_dgl(
        drug_cnt, target_cnt,
        adjs=pos_adj_dict,
        relation2id=relation2id
    ), ssp_multigraph_to_dgl(
        drug_cnt, target_cnt,
        adjs=neg_adj_dict,
        relation2id=relation2id
    )

    print("train positive graph: ", train_pos_graph[0])
    print("train negative graph: ", train_neg_graph[0])
    params.rel2id = train_pos_graph[1]
    params.num_rels = len(params.rel2id)
    params.id2rel = {
        v: train_pos_graph[0].to_canonical_etype(k) for k, v in params.rel2id.items()
    }
    train_pos_graph, train_neg_graph = train_pos_graph[0], train_neg_graph[0]
    # add edges in valid set, whose 'mask' = 1

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.intra_enc1 = 'afp'
    params.intra_enc2 = 'afp'  # 'rnn'
    params.loss = 'focal'
    params.task = 'dt'  # options: 'dd', 'all'
    params.is_test = False

    small_intra_g_list = [small_mol_graphs[id2drug[i]][1] for i in range(small_cnt)]
    target_intra_g_list = [macro_mol_graphs[id2target[i]][2] for i in range(target_cnt)]

    # init molecule graphs for all molecules in inter_graph
    mol_graphs = {
        'small': small_intra_g_list,  # d_id = idx
        'target': target_intra_g_list  # t_id = idx
    }
    if small_cnt < drug_cnt:
        mol_graphs['bio'] = [macro_mol_graphs[id2drug[i]][2] for i in
                             range(small_cnt, drug_cnt)]  # d_id = small_cnt+idx

    model_file_name = f'model_{params.intra_enc1}_{params.intra_enc2}_{params.dataset}_{params.loss}.model'
    result_file_name = f'result_{params.intra_enc1}_{params.intra_enc2}_{params.dataset}_{params.loss}.csv'

    if params.dataset != 'full_drugbank':
        if params.task == 'dt':
            params.pred_rels = dt_types
        else:
            raise NotImplementedError
    else:
        if params.task == 'dt':
            params.pred_rels = dt_types
        elif params.task == 'dd':
            raise NotImplementedError
            # todo
        elif params.task == 'all':
            params.pred_rels = None

    if not params.is_test:
        # init model, opt, loss
        encoder, decoder, dgi_model = initialize_BioMIP(params)
        opt = build_optimizer(encoder, decoder, dgi_model, params)
        loss_fn = select_loss_function(params.loss)  # a loss dict 'loss name': (weight, loss_fuction)
        best_auc = 0.
        best_epoch = -1
        for epoch in range(1, params.n_epoch + 1):
            emb_intra, emb_inter = train(encoder, decoder, dgi_model,
                                         opt, loss_fn,
                                         mol_graphs,
                                         train_pos_graph, train_neg_graph,
                                         pred_rels=params.pred_rels)
            print('predicting for valid data')
            val_G, val_P = predicting(encoder,
                                      emb_intra, emb_inter,
                                      triplets['valid']['pos'], triplets['valid']['neg'])
            val = metrics.roc_auc_score(val_G, val_P)
            print(f'valid AUROC: ', val)
            if val > best_auc:
                best_auc = val
                best_epoch = epoch
                torch.save(encoder.state_dict(), f'trained_models/{model_file_name})')
                print(f'AUROC improved at epoch {best_epoch}')
                print(f'val AUROC {val}, AUPRC {metrics.average_precision_score(val_G, val_P)}, '
                      f'F1: {metrics.f1_score(val_G, eval_threshold(val_G, val_P)[1])}')
                print('predicting for test data')
                test_G, test_P = predicting(decoder,
                                            emb_intra, emb_inter,
                                            triplets['test']['pos'], triplets['test']['neg'])
                auroc, auprc, f1 = metrics.roc_auc_score(test_G, test_P), \
                                   metrics.average_precision_score(test_G, test_P), \
                                   metrics.f1_score(test_G, eval_threshold(val_G, val_P))
                if not os.path.exists(f'results/{result_file_name}'):
                    with open(f'results/{result_file_name}', 'w') as f:
                        f.write('time,dataset,auroc,auprc,f1\n')
                with open(f'results/{result_file_name}', 'a+') as f:
                    f.write(f'{time_str},{params.dataset},{auroc},{auprc},{f1}')

import logging
import os
import time

import pandas as pd
import torch
from sklearn import metrics

from model import build_optimizer
from model.customized_loss import select_loss_function
from model.model_config import initialize_BioMIP
from utils.arg_parser import parser
from utils.data_utils import eval_threshold
from utils.generate_intra_graph_db import generate_small_mol_graph_datasets, generate_macro_mol_graph_datasets
from utils.hete_data_utils import ssp_multigraph_to_dgl, \
    dd_dt_tt_build_inter_graph_from_links, build_valid_test_graph
from utils.intra_graph_dataset import IntraGraphDataset
# training fuction on each epoch
from utils.utils import calc_aupr


def train(encoder, decoder_intra, decoder_inter, dgi_model,
          opt, loss_fn: dict,
          mol_graphs,
          train_pos_graph, train_neg_graph):
    encoder.train()
    decoder_intra.train()
    decoder_inter.train()
    dgi_model.train()
    opt.zero_grad()
    # emb_intra: dict, keys: "small", "bio", "target"
    # emb_inter: dict, keys: "drug", "target"
    emb_intra, emb_inter = encoder(mol_graphs,
                                   train_pos_graph)
    # print("emb_intra", emb_intra)
    # print("emb_inter", emb_inter)
    dgi_loss = dgi_model(emb_intra, emb_inter,
                         train_pos_graph, train_neg_graph)

    # decoder:  return dict
    if emb_intra['bio'] is not None:
        emb_intra = torch.cat((emb_intra['small'], emb_intra['bio']), dim=0)
    else:
        emb_intra = emb_intra['small']
    # emb_intra = emb_intra['small']
    emb_inter = emb_inter['drug']
    pos_scores_intra = torch.hstack(tuple(decoder_intra(train_pos_graph, emb_intra).values()))
    neg_scores_intra = torch.hstack(tuple(decoder_intra(train_neg_graph, emb_intra).values()))
    labels = torch.cat([torch.ones(pos_scores_intra.shape[0]), torch.zeros(neg_scores_intra.shape[0])])  # .numpy()
    pos_scores_inter = torch.hstack(tuple(decoder_inter(train_pos_graph, emb_inter).values()))
    neg_scores_inter = torch.hstack(tuple(decoder_inter(train_neg_graph, emb_inter).values()))
    score_inter, score_intra = torch.cat([pos_scores_inter, neg_scores_inter]), torch.cat(
        [pos_scores_intra, neg_scores_intra])
    # print(score_inter, score_intra, labels)
    BCEloss = torch.mean(loss_fn['ERROR'](score_inter, labels))
    # BCEloss += torch.mean(loss_fn['ERROR'](score_intra, labels))
    KLloss = torch.mean(loss_fn['DIFF'](score_intra, score_inter))
    # KLloss = torch.mean(loss_fn['DIFF'](score_intra, torch.log(score_inter)))

    curr_loss = params.alpha_loss * BCEloss + params.beta_loss * KLloss + params.gamma_loss * dgi_loss
    curr_loss.backward()
    print("back:", BCEloss.item(), KLloss.item(), dgi_loss.item())
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


def predicting(model_intra, model_inter,
               intra_feats, inter_feats,
               pos_g, neg_g,
               multi_res=False):
    if multi_res:
        pos_pred, neg_pred = model_inter(pos_g, inter_feats), model_inter(neg_g, inter_feats)
        res = {}
        for k in pos_g.canonical_etypes[:-2]:
            res[k] = (torch.cat([torch.ones(pos_pred[k].shape[0]), torch.zeros(neg_pred[k].shape[0])]).numpy(),
                      torch.cat([pos_pred[k], neg_pred[k]]).detach().numpy())
        return res

    pos_score1 = torch.hstack(tuple(model_intra(pos_g, intra_feats).values()))
    neg_score1 = torch.hstack(tuple(model_intra(neg_g, intra_feats).values()))
    pos_score2 = torch.hstack(tuple(model_inter(pos_g, inter_feats).values()))
    neg_score2 = torch.hstack(tuple(model_inter(neg_g, inter_feats).values()))
    labels = torch.cat([torch.ones(pos_score1.shape[0]), torch.zeros(neg_score1.shape[0])]).numpy()
    return labels, torch.cat([pos_score1, neg_score1]).detach().numpy(), \
           torch.cat([pos_score2, neg_score2]).detach().numpy()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    # initialize_experiment(params)
    # save featurized intra-view graphs
    print(params)

    time_str = time.strftime('%m-%d_%H_%M', time.localtime(time.time()))

    params.aln_path = '/data/rzy/drugbank_prot/full_drugbank/aln'
    params.npy_path = '/data/rzy/drugbank_prot/full_drugbank/pconsc4'

    if params.dataset == 'full':
        params.small_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'deep':
        params.small_mol_db_path = f'/data/rzy/deep/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/deep/prot_graph_db'  # _{params.prot_featurizer}
    else:
        raise NotImplementedError

    # macro_mol_list = pd.read_csv(f'data/{params.dataset}/macro_seqs.csv', header=None, names=['id', '_'])[
    #     'id'].tolist()
    # macro_mol_list = [str(i) for i in macro_mol_list]  # assure mol_id is a string

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

    # print("train positive graph: ", train_pos_graph[0])
    # print("train negative graph: ", train_neg_graph[0])

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

    model_file_name = f'{time_str}model_{params.intra_enc1}_{params.intra_enc2}_{params.dataset}{params.split}_{params.loss}.model'
    result_file_name = f'{time_str}_result_{params.intra_enc1}_{params.intra_enc2}_{params.dataset}{params.split}_{params.loss}.csv'

    if not params.is_test:
        # init model, opt, loss
        encoder, decoder_intra, decoder_inter, dgi_model = initialize_BioMIP(params)
        opt = build_optimizer(encoder, decoder_intra, decoder_inter, dgi_model, params)
        loss_fn = select_loss_function(params.loss)  # a loss dict 'loss name': (weight, loss_fuction)
        best_auc = 0.
        best_epoch = -1
        # print(id2relation, relation2id)
        vgp, vgn = build_valid_test_graph(
            drug_cnt,
            edges=triplets['valid']['pos'],
            relation2id=relation2id,
            id2relation=id2relation
        ), build_valid_test_graph(
            drug_cnt,
            edges=triplets['valid']['neg'],
            relation2id=relation2id,
            id2relation=id2relation
        )
        tgp, tgn = build_valid_test_graph(
            drug_cnt,
            edges=triplets['test']['pos'],
            relation2id=relation2id,
            id2relation=id2relation
        ), build_valid_test_graph(
            drug_cnt,
            edges=triplets['test']['neg'],
            relation2id=relation2id,
            id2relation=id2relation
        )
        for epoch in range(1, params.n_epoch + 1):
            emb_intra, emb_inter = train(encoder, decoder_intra, decoder_inter, dgi_model,
                                         opt, loss_fn,
                                         mol_graphs,
                                         train_pos_graph, train_neg_graph)
            print('predicting for valid data')

            val_G, val_P1, val_P2 = predicting(decoder_intra, decoder_inter,
                                               emb_intra, emb_inter,
                                               vgp, vgn)
            val1 = metrics.roc_auc_score(val_G, val_P1)
            # print('-----',val_P2)
            val2 = metrics.roc_auc_score(val_G, val_P2)
            print(f'valid AUROC: ', val1, val2)
            if val2 > best_auc:
                best_auc = val2
                best_epoch = epoch
                torch.save(encoder.state_dict(), f'trained_models/encoder_{model_file_name})')
                torch.save(decoder_inter.state_dict(), f'trained_models/interdec_{model_file_name})')
                torch.save(decoder_intra.state_dict(), f'trained_models/interdec_{model_file_name})')
                print(f'AUROC improved at epoch {best_epoch}')
                print(
                    f'val AUROC {val2}, AP {metrics.average_precision_score(val_G, val_P2)} F1: {metrics.f1_score(val_G, eval_threshold(val_G, val_P2)[1])}')
                print('predicting for test data')
                test_G, test_P1, test_P2 = predicting(decoder_intra, decoder_inter,
                                                      emb_intra, emb_inter,
                                                      tgp, tgn)
                # AP

                test_auroc, test_ap, test_auprc, test_f1 = metrics.roc_auc_score(test_G, test_P2), \
                                                           metrics.average_precision_score(test_G, test_P2), \
                                                           calc_aupr(test_G, test_P2), \
                                                           metrics.f1_score(test_G, eval_threshold(test_G, test_P2)[1])
                if not os.path.exists(f'results/{result_file_name}'):
                    with open(f'results/{result_file_name}', 'w') as f:
                        f.write('epoch,auroc,ap,auprc,f1\n')
                with open(f'results/{result_file_name}', 'a+') as f:
                    f.write(f'{epoch},{test_auroc},{test_ap},{test_auprc},{test_f1}\n')

        # calculate and sace the final test results
        encoder.load_state_dict(torch.load(f'trained_models/encoder_{model_file_name})'))
        decoder_inter.load_state_dict(torch.load(f'trained_models/interdec_{model_file_name})'))
        decoder_intra.load_state_dict(torch.load(f'trained_models/interdec_{model_file_name})'))
        emb_intra, emb_inter = encoder(mol_graphs, train_pos_graph)
        rel2gt_pred = predicting(decoder_intra, decoder_inter,
                                 emb_intra, emb_inter,
                                 tgp, tgn,
                                 multi_res=True)

        # parse the rel_id to raw relation type for comparison to baselines
        # rel,
        res_list = []
        print(len(id2relation), id2relation)
        print(relation2id)
        print(len(rel2gt_pred.keys()), rel2gt_pred.keys())
        total_len, tot_auroc, tot_auprc, tot_ap, tot_f1 = 0, 0.0, 0.0, 0.0, 0.0
        for k, v in rel2gt_pred.items():
            print(id2relation[int(k)])
            _len = v[0].shape[0]
            total_len += _len
            test_auroc, test_ap, test_auprc, test_f1 = metrics.roc_auc_score(v[0], v[1]), \
                                                       metrics.average_precision_score(v[0], v[1]), \
                                                       calc_aupr(v[0], v[1]), \
                                                       metrics.f1_score(v[0], eval_threshold(v[0], v[1])[1])
            tot_auroc += test_auroc * _len
            tot_auprc += test_auprc * _len
            tot_ap += test_ap * _len
            tot_f1 += test_f1 * _len
            res_list.append([int(k), test_auroc, test_auprc, test_ap, test_f1])

        pd.DataFrame(res_list).to_csv(f'results/{result_file_name}_multi.csv',
                                      header=['rel_name', 'auroc', 'auprc', 'ap', 'f1'])
        with open(f'results/{result_file_name}', 'a+') as f:
            f.write(
                f'final,{params.dataset},'
                f'{tot_auroc / total_len},'
                f'{tot_auprc / total_len},'
                f'{tot_ap / total_len},'
                f'{tot_f1 / total_len}\n'
            )

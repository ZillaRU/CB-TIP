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
from utils.utils import calc_aupr


def train(encoder, decoder_intra, decoder_inter, dgi_model,
          opt, loss_fn: dict,
          mol_graphs,
          train_pos_graph, train_neg_graph,
          loss_save_file,
          epo_no,
          intra_pairs,
          epo_num_with_fp=200):
    # print(params.learning_rate)
    encoder.train()
    decoder_intra.train()
    decoder_inter.train()
    dgi_model.train()
    opt.zero_grad()
    emb_intra, emb_inter = encoder(mol_graphs,
                                   train_pos_graph)
    dgi_loss = dgi_model(emb_intra, emb_inter,
                         train_pos_graph, train_neg_graph)
    emb_intra_prot = emb_intra['target']
    if emb_intra['bio'] is not None:
        emb_intra_prot = torch.cat((emb_intra['target'], emb_intra['bio']), dim=0)
        emb_intra = torch.cat((emb_intra['small'], emb_intra['bio']), dim=0)
    else:
        emb_intra = emb_intra['small']
    # emb_intra = emb_intra['small']
    emb_inter = emb_inter['drug']
    pos_scores_intra = torch.hstack(tuple(decoder_intra(train_pos_graph, emb_intra).values()))
    neg_scores_intra = torch.hstack(tuple(decoder_intra(train_neg_graph, emb_intra).values()))
    labels = torch.cat(
        [torch.ones(pos_scores_intra.shape[0]), torch.zeros(neg_scores_intra.shape[0])]).cuda()  # .numpy()
    pos_scores_inter = torch.hstack(tuple(decoder_inter(train_pos_graph, emb_inter).values()))
    neg_scores_inter = torch.hstack(tuple(decoder_inter(train_neg_graph, emb_inter).values()))

    score_inter, score_intra = torch.cat([pos_scores_inter, neg_scores_inter]), torch.cat(
        [pos_scores_intra, neg_scores_intra])
    BCEloss = torch.mean(loss_fn['ERROR'](score_inter, labels))
    BCEloss += params.alpha_loss * torch.mean(loss_fn['ERROR'](score_intra, labels))
    KLloss = torch.mean(loss_fn['DIFF'](score_intra, score_inter))
    if epo_no < epo_num_with_fp:
        emb_intra_in1, emb_intra_in2 = emb_intra[intra_pairs['in'][0]], emb_intra[intra_pairs['in'][1]]
        emb_intra_bt2 = emb_intra[intra_pairs['bt'][1]]
        diff_loss = loss_fn['FP_INTRA'](emb_intra_in1, emb_intra_in2, emb_intra_bt2)
                    # + loss_fn['FP_INTRA'](prot_emb_intra_in1, prot_emb_intra_in2, prot_emb_intra_bt2)
        print("back:", BCEloss.item(), KLloss.item(), dgi_loss.item(), diff_loss.item())
        curr_loss = BCEloss + params.beta_loss * KLloss + params.gamma_loss * dgi_loss + params.theta_loss * diff_loss
    else:
        print("back:", BCEloss.item(), KLloss.item(), dgi_loss.item())
        curr_loss = BCEloss + params.beta_loss * KLloss + params.gamma_loss * dgi_loss
    curr_loss.backward()

    opt.step()
    print('Train epoch: {} Loss: {:.6f}'.format(epoch, curr_loss.item()))
    with open(f'results/{loss_save_file}', 'a+') as f:
        f.write(f'{BCEloss.item()},{KLloss.item()},{dgi_loss.item()},{curr_loss.item()}\n')
    return emb_intra, emb_inter


def predicting(model_intra, model_inter,
               intra_feats, inter_feats,
               pos_g, neg_g,
               multi_res=False):
    model_intra.eval()
    model_inter.eval()
    if multi_res:
        pos_pred, neg_pred = model_inter(pos_g, inter_feats), model_inter(neg_g, inter_feats)
        res = {}
        for k in pos_g.canonical_etypes[:-2]:
            res[k] = (torch.cat([torch.ones(pos_pred[k].shape[0]), torch.zeros(neg_pred[k].shape[0])]).cpu().numpy(),
                      torch.cat([pos_pred[k], neg_pred[k]]).detach().cpu().numpy())
        return res

    pos_score1 = torch.hstack(tuple(model_intra(pos_g, intra_feats).values()))
    neg_score1 = torch.hstack(tuple(model_intra(neg_g, intra_feats).values()))
    pos_score2 = torch.hstack(tuple(model_inter(pos_g, inter_feats).values()))
    neg_score2 = torch.hstack(tuple(model_inter(neg_g, inter_feats).values()))
    labels = torch.cat([torch.ones(pos_score1.shape[0]), torch.zeros(neg_score1.shape[0])]).numpy()
    return labels, torch.cat([pos_score1, neg_score1]).detach().cpu().numpy(), \
           torch.cat([pos_score2, neg_score2]).detach().cpu().numpy()


def save_id_mapping(dataset, split,
                    id2drug, drug2id,
                    id2target, target2id,
                    id2relation):
    np.save(f'data/{dataset}/{split}_id2drug.npy', id2drug)
    np.save(f'data/{dataset}/{split}_drug2id.npy', drug2id)
    np.save(f'data/{dataset}/{split}_id2target.npy', id2target)
    np.save(f'data/{dataset}/{split}_target2id.npy', target2id)
    np.save(f'data/{dataset}/{split}_id2relation.npy', id2relation)


def load_fp_contrastive_pairs(dataset, split):
    drug2id = np.load(f'data/{dataset}/{split}_drug2id.npy', allow_pickle=True).item()
    target2id = np.load(f'data/{dataset}/{split}_target2id.npy', allow_pickle=True).item()

    df = pd.read_csv(f'data/{dataset}/in_pairs.csv', names=['mol1', 'mol2'])
    df['mol1'] = df['mol1'].map(drug2id)
    df['mol2'] = df['mol2'].map(drug2id)
    df.dropna(inplace=True)
    # assert df.dropna().shape == df.shape
    pos_mols1, pos_mols2 = df['mol1'].tolist(), df['mol2'].tolist()

    df = pd.read_csv(f'data/{dataset}/bt_pairs.csv', names=['mol1', 'mol2'])
    df['mol1'] = df['mol1'].map(drug2id)
    df['mol2'] = df['mol2'].map(drug2id)
    df.dropna(inplace=True)
    neg_mols1, neg_mols2 = df['mol1'].tolist(), df['mol2'].tolist()

    # emb_intra = concat target + bio    emb_inter = small + bio
    if dataset == 'full':
        for drug in drug2id.keys():
            if drug.startswith('DB'):
                target2id[drug] = drug2id[drug] + (target_cnt - small_cnt)

    prot_df = pd.read_csv(f'data/{dataset}/in_pairs_prot.csv', names=['mol1', 'mol2'])
    prot_df['mol1'] = prot_df['mol1'].map(target2id)
    prot_df['mol2'] = prot_df['mol2'].map(target2id)
    prot_df.dropna(inplace=True)
    pos_prots1, pos_prots2 = prot_df['mol1'].tolist(), prot_df['mol2'].tolist()

    prot_df = pd.read_csv(f'data/{dataset}/bt_pairs_prot.csv', names=['mol1', 'mol2'])
    prot_df['mol1'] = prot_df['mol1'].map(target2id)
    prot_df['mol2'] = prot_df['mol2'].map(target2id)
    prot_df.dropna(inplace=True)
    neg_prots1, neg_prots2 = prot_df['mol1'].tolist(), prot_df['mol2'].tolist()
    return {
        'in': [pos_mols1, pos_mols2],
        'bt': [neg_mols1, neg_mols2],
        'in_prot': [pos_prots1, pos_prots2],
        'bt_prot': [neg_prots1, neg_prots2]
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    params = parser.parse_args()
    print(params)

    time_str = time.strftime('%m-%d_%H_%M', time.localtime(time.time()))

    params.aln_path = '/data/rzy/drugbank_prot/full_drugbank/aln'
    params.npy_path = '/data/rzy/drugbank_prot/full_drugbank/pconsc4'

    if params.dataset == 'cb-db':
        params.small_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/drugbank_prot/{params.dataset}/prot_graph_db'  # _{params.prot_featurizer}
    elif params.dataset == 'c-db':
        params.small_mol_db_path = f'/data/rzy/deep/smile_graph_db_{params.SMILES_featurizer}'
        params.macro_mol_db_path = f'/data/rzy/deep/prot_graph_db'  # _{params.prot_featurizer}
    else:
        raise NotImplementedError

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

    params.rel2id = train_pos_graph[1]
    params.num_rels = len(params.rel2id)
    params.id2rel = {
        v: train_pos_graph[0].to_canonical_etype(k) for k, v in params.rel2id.items()
    }
    train_pos_graph, train_neg_graph = train_pos_graph[0], train_neg_graph[0]
    # add edges in valid set, whose 'mask' = 1

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        train_pos_graph = train_pos_graph.to(params.device)
        train_neg_graph = train_neg_graph.to(params.device)
    else:
        params.device = torch.device('cpu')

    params.intra_enc1 = 'afp'
    params.intra_enc2 = 'afp'  # 'rnn'
    params.loss = 'focal'
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

    model_file_name = f'{time_str}model_{params.dataset}{params.split}.model' \
        if params.model_filename is None else params.model_filename
    result_file_name = f'{time_str}result_{params.dataset}{params.split}.csv'
    encoder, decoder_intra, decoder_inter, ff_contra_net = initialize_BioMIP(params)
    tgp, tgn = build_valid_test_graph(
        drug_cnt,
        edges=triplets['test']['pos'],
        relation2id=relation2id,
        id2relation=id2relation
    ).to(params.device), build_valid_test_graph(
        drug_cnt,
        edges=triplets['test']['neg'],
        relation2id=relation2id,
        id2relation=id2relation
    ).to(params.device)
    if not params.is_test:
        # init opt, loss
        opt = build_optimizer(encoder, decoder_intra, decoder_inter, ff_contra_net, params)
        loss_fn = select_loss_function(params.loss)  # a loss dict 'loss name': (weight, loss_fuction)
        best_auc = 0.
        best_epoch = -1
        vgp, vgn = build_valid_test_graph(
            drug_cnt,
            edges=triplets['valid']['pos'],
            relation2id=relation2id,
            id2relation=id2relation
        ).to(params.device), build_valid_test_graph(
            drug_cnt,
            edges=triplets['valid']['neg'],
            relation2id=relation2id,
            id2relation=id2relation
        ).to(params.device)
        cnt_trival = 0
        intra_pairs = load_fp_contrastive_pairs(params.dataset, params.split)
        for epoch in range(1, params.max_epoch + 1):
            emb_intra, emb_inter = train(encoder, decoder_intra, decoder_inter, dgi_model,
                                         opt, loss_fn,
                                         mol_graphs,
                                         train_pos_graph, train_neg_graph,
                                         f'09_loss_{result_file_name}',
                                         epoch,
                                         intra_pairs,
                                         epo_num_with_fp=20)
            print('predicting for valid data')

            val_G, val_P1, val_P2 = predicting(decoder_intra, decoder_inter,
                                               emb_intra, emb_inter,
                                               vgp, vgn)
            val1 = metrics.roc_auc_score(val_G, val_P1)
            val2 = metrics.roc_auc_score(val_G, val_P2)
            print(f'valid AUROC: ', val1, val2)
            if val2 > best_auc:
                cnt_trival = 0
                best_auc = val2
                best_epoch = epoch
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
                        f.write('epoch,auroc,auprc,ap,f1\n')
                with open(f'results/{result_file_name}', 'a+') as f:
                    f.write(f'{epoch},{test_auroc},{test_auprc},{test_ap},{test_f1}\n')
            else:
                if epoch > params.min_epoch:
                    cnt_trival += 1
                if cnt_trival >= params.stop_thresh:
                    print('Stop training due to more than 20 epochs with no improvement')
                    break
    encoder.load_state_dict(torch.load(f'trained_models/encoder_{model_file_name}'))
    decoder_inter.load_state_dict(torch.load(f'trained_models/interdec_{model_file_name}'))

    decoder_intra.load_state_dict(torch.load(f'trained_models/intradec_{model_file_name}'))
    emb_intra, emb_inter = encoder(mol_graphs, train_pos_graph)

    if emb_intra['bio'] is not None:
        # np.savetxt(f'emb_vis/{params.dataset}_intra_bio_for_visual.csv', emb_intra['bio'].detach().cpu().numpy(), delimiter=',')
        emb_intra = torch.cat((emb_intra['small'], emb_intra['bio']), dim=0)

    else:
        emb_intra = emb_intra['small']

    emb_inter = emb_inter['drug']
    rel2gt_pred = predicting(decoder_intra, decoder_inter,
                             emb_intra, emb_inter,
                             tgp, tgn,
                             multi_res=True)
    res_list = []

    print(len(rel2gt_pred.keys()), rel2gt_pred.keys())  # 62
    total_len, tot_auroc, tot_auprc, tot_ap, tot_f1 = 0, 0.0, 0.0, 0.0, 0.0
    for k, v in rel2gt_pred.items():
        _len = v[0].shape[0]
        if _len == 0:
            res_list.append([int(k[1]), 0, 0, 0, 0, 0])
            continue
        try:
            test_auroc, test_ap, test_auprc, test_f1 = metrics.roc_auc_score(v[0], v[1]), \
                                                       metrics.average_precision_score(v[0], v[1]), \
                                                       calc_aupr(v[0], v[1]), \
                                                       metrics.f1_score(v[0], eval_threshold(v[0], v[1])[1])
            total_len += _len
        except ValueError:
            res_list.append([int(k[1]),0, 0, 0, 0, 0])
            continue
        tot_auroc += test_auroc * _len
        tot_auprc += test_auprc * _len
        tot_ap += test_ap * _len
        tot_f1 += test_f1 * _len
        res_list.append([int(k[1]), _len, test_auroc, test_auprc, test_ap, test_f1])
    pd.DataFrame(res_list).to_csv(f'results/{result_file_name}_multi.csv', index=False,
                                  header=['rel_name', 'test_len', 'auroc', 'auprc', 'ap', 'f1'])
    with open(f'results/{result_file_name}', 'a+') as f:
        f.write(
            f'final,{params.dataset},'
            f'{tot_auroc / total_len},'
            f'{tot_auprc / total_len},'
            f'{tot_ap / total_len},'
            f'{tot_f1 / total_len}\n'
        )

# python train_main.py -d deep -sp 415-2 --gpu 2 --beta_loss 0.3 --gamma_loss 0.001 --alpha_loss 1 -lr 0.001 --max_epoch 500

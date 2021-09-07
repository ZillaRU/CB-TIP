import logging
import os
import struct
from functools import partial

import dgl
import lmdb
import numpy as np
import pandas as pd
import rdkit.Chem as chem
import torch
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer, BaseAtomFeaturizer, \
    ConcatFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons, \
    atom_hybridization_one_hot, atom_total_num_H_one_hot, BaseBondFeaturizer, one_hot_encoding, \
    AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

from utils.data_utils import serialize
from utils.mol_utils import PSSM_calculation, pro_res_table, pro_res_aliphatic_table, pro_res_polar_neutral_table, \
    pro_res_acidic_charged_table, pro_res_basic_charged_table, pro_res_aromatic_table, res_pkb_table, res_pkx_table, \
    res_hydrophobic_ph7_table, res_pka_table, res_hydrophobic_ph2_table, res_weight_table, res_pl_table, \
    one_of_k_encoding


# , PAGTNAtomFeaturizer, PAGTNEdgeFeaturizer


def build_molecule_graph(args_, graph_mode='bigraph', featurizer='base'):
    """Construct graphs from SMILES and featurize them
    options:
    BaseAtomFeaturizer
    CanonicalAtomFeaturizer     CanonicalBondFeaturizer
    PretrainAtomFeaturizer      WeaveEdgeFeaturizer
    WeaveAtomFeaturizer         PretrainBondFeaturizer
    AttentiveFPAtomFeaturizer   AttentiveFPBondFeaturizer

    Returns
    -------
    DGLGraph
        a graph constructed and featurized or None
        parsed by RDKit
    """
    # idx, (mol_id, smiles) = args_
    mol_id, smiles = args_
    mol = chem.MolFromSmiles(smiles)
    if graph_mode == 'bigraph' and featurizer == 'base':
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=PretrainAtomFeaturizer(),
                           edge_featurizer=PretrainBondFeaturizer(),
                           canonical_atom_order=False)
    elif graph_mode == 'bigraph' and featurizer == 'afp':  # Attentive FP
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=AttentiveFPAtomFeaturizer(atom_data_field='hv'),
                           edge_featurizer=AttentiveFPBondFeaturizer(bond_data_field='he')
                           )
    else:  # custom
        def chirality(atom):
            try:
                return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                       [atom.HasProp('_ChiralityPossible')]
            except:
                return [False, False] + [atom.HasProp('_ChiralityPossible')]

        atom_featurizer = BaseAtomFeaturizer(
            {'hv': ConcatFeaturizer([
                partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                        encode_unknown=True),
                partial(atom_degree_one_hot, allowable_set=list(range(6))),
                atom_formal_charge, atom_num_radical_electrons,
                partial(atom_hybridization_one_hot, encode_unknown=True),
                lambda atom: [0],  # A placeholder for aromatic information,
                atom_total_num_H_one_hot, chirality
            ],
            )})
        bond_featurizer = BaseBondFeaturizer({
            'he': lambda bond: [0 for _ in range(10)]
        })
        g = mol_to_bigraph(mol,
                           add_self_loop=True,
                           node_featurizer=atom_featurizer,
                           edge_featurizer=bond_featurizer
                           )
    if g is None: return (-1, -1)
    # print(g.node_attr_schemes())
    n_node, n_edge = g.num_nodes(), g.num_edges()
    _tp = [torch.as_tensor(g.ndata[k].view(n_node, -1), dtype=torch.float32) for k in g.node_attr_schemes().keys()]
    g.ndata['nfeats'] = torch.cat(tuple(_tp), 1)
    _tp = [torch.as_tensor(g.edata[k].view(n_edge, -1), dtype=torch.float32) for k in g.edge_attr_schemes().keys()]
    g.edata['efeats'] = torch.cat(tuple(_tp), 1)

    datum = {
        # 'mol_id': mol_id,
        'mol_graph': g,
        'graph_size': g.num_nodes()
    }
    # idx = '{:08}'.format(idx).encode('ascii')
    idx = str(mol_id).encode('ascii')
    return (idx, datum)


def init_folder(params):
    global _dataset, _aln_path, _npy_path
    _dataset, _aln_path, _npy_path = params.dataset, params.aln_path, params.npy_path


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        temp = one_of_k_encoding(pro_seq[i], pro_res_table)
        if temp != -1:
            pro_hot[i,] = temp
        else:
            return None
        # pro_hot[i,] = one_of_k_encoding_unk(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def macro_mol_feature(aln, seq):
    pssm = PSSM_calculation(aln, seq)
    other_feature = seq_feature(seq)
    if other_feature is None:
        return None
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


def macro_mol_to_feature(seq, aln):
    feature = macro_mol_feature(aln, seq)
    return feature if feature is not None else None


def build_seq_to_graph(args):
    mol_id, seq = args
    aln_path = f"{_aln_path}/{mol_id}.aln"
    cmap_path = f'{_npy_path}/{mol_id}.npy'
    if not os.path.exists(aln_path) or not os.path.exists(cmap_path):
        print(mol_id, aln_path, cmap_path)
        return (-1, -1)
    macro_mol_edge_index = []
    contact_map = np.load(cmap_path, allow_pickle=True)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        macro_mol_edge_index.append([i, j])
    feat = macro_mol_to_feature(seq, aln_path)
    if feat is None:
        return (-1, -1)
    mol_feature = torch.from_numpy(feat)
    g = dgl.graph((torch.from_numpy(index_row), torch.from_numpy(index_col)))
    # g = dgl.heterograph((index_row, index_col))
    print(g.num_nodes(), mol_feature.shape)
    print(mol_id)
    g.ndata['nfeats'] = torch.tensor(mol_feature, dtype=torch.float32)
    g.edata['efeats'] = torch.ones([g.num_edges(), 1], dtype=torch.float32)  # todo
    datum = {
        'seq': seq,
        'mol_graph': g
    }

    idx = mol_id.encode('ascii')
    return (idx, datum)


def generate_small_mol_graph_datasets(params):
    dbname = 'small_mol'
    logging.info(f"Construct intra-view graphs for small molecules in {params.dataset}...")

    SMILES_csv = pd.read_csv(f'data/{params.dataset}/SMILESstrings.csv', header=None)

    # todo: fix map_size
    env = lmdb.open(params.small_mol_db_path, map_size=1e9, max_dbs=6)

    num_mol = SMILES_csv.shape[0]

    with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
        txn.put('num_graphs'.encode(), num_mol.to_bytes(int.bit_length(num_mol), byteorder='little'))

    graph_sizes = []
    seq_list = np.array(SMILES_csv).tolist()
    # get node_feature_dimension and edge_feature_dimension
    _, g_datum = build_molecule_graph((seq_list[0][0], seq_list[0][1]))
    temp_g = g_datum['mol_graph']
    node_feat_dim = temp_g.ndata['nfeats'].shape[1]
    edge_feat_dim = temp_g.edata['efeats'].shape[1]
    logging.info(f'node_feat_dim = {node_feat_dim}, edge_feat_dim = {edge_feat_dim}')

    for [_id, _val] in seq_list:
        idx, datum = build_molecule_graph((_id, _val))
        if idx == -1:
            print(_id)
            continue
        graph_sizes.append(datum['graph_size'])
        with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
            txn.put(idx, serialize(datum))

    with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
        print('==== ==== ==== ==== ==== ==== ==== Writing ==== ==== ==== ==== ==== ==== ====')
        bit_len = int.bit_length(node_feat_dim)
        txn.put('node_feat_dim'.encode(), (int(node_feat_dim)).to_bytes(bit_len, byteorder='little'))
        txn.put('edge_feat_dim'.encode(), (int(edge_feat_dim)).to_bytes(bit_len, byteorder='little'))
        txn.put('avg_molgraph_size'.encode(), struct.pack('f', float(np.mean(graph_sizes))))
        txn.put('min_molgraph_size'.encode(), struct.pack('f', float(np.min(graph_sizes))))
        txn.put('max_molgraph_size'.encode(), struct.pack('f', float(np.max(graph_sizes))))
        txn.put('std_molgraph_size'.encode(), struct.pack('f', float(np.std(graph_sizes))))


def generate_macro_mol_graph_datasets(params):
    dbname = 'macro_mol'
    logging.info(f"Construct intra-view graphs for macro molecules in {params.dataset}...")
    seq_list = pd.read_csv(f'data/{params.dataset}/target_seqs.csv', header=None).values.tolist()
    if params.dataset == 'full':
        seq_list += pd.read_csv(f'data/{params.dataset}/biotech_seqs.csv', header=None).values.tolist()

    # todo: fix map_size
    env = lmdb.open(params.macro_mol_db_path, map_size=1e9, max_dbs=6)

    num_mol = len(seq_list)

    with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
        txn.put('num_graphs'.encode(), num_mol.to_bytes(int.bit_length(num_mol), byteorder='little'))

    graph_sizes = []

    # get node_feature_dimension and edge_feature_dimension
    init_folder(params)
    _, g_datum = build_seq_to_graph((seq_list[0][0], seq_list[0][1]))
    temp_g = g_datum['mol_graph']
    node_feat_dim = temp_g.ndata['nfeats'].shape[1]
    edge_feat_dim = temp_g.edata['efeats'].shape[1]
    logging.info(f'node_feat_dim = {node_feat_dim}, edge_feat_dim = {edge_feat_dim}')

    for _id, _val in seq_list:
        idx, datum = build_seq_to_graph((_id, _val))
        if idx == -1:
            print(_id)
            continue
        graph_sizes.append(len(datum['seq']))
        with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
            txn.put(idx, serialize(datum))

    with env.begin(write=True, db=env.open_db(dbname.encode())) as txn:
        print('==== ==== ==== ==== ==== ==== Writing ==== ==== ==== ==== ==== ====')
        bit_len = int.bit_length(node_feat_dim)
        txn.put('node_feat_dim'.encode(), (int(node_feat_dim)).to_bytes(bit_len, byteorder='little'))
        txn.put('edge_feat_dim'.encode(), (int(edge_feat_dim)).to_bytes(bit_len, byteorder='little'))
        txn.put('avg_molgraph_size'.encode(), struct.pack('f', float(np.mean(graph_sizes))))
        txn.put('min_molgraph_size'.encode(), struct.pack('f', float(np.min(graph_sizes))))
        txn.put('max_molgraph_size'.encode(), struct.pack('f', float(np.max(graph_sizes))))
        txn.put('std_molgraph_size'.encode(), struct.pack('f', float(np.std(graph_sizes))))

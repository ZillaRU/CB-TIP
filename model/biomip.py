import dgl
import torch
import torch.nn as nn

from model.inter_gnn.dgl_rgcn import RGCN
from model.intra_GNN import Intra_AttentiveFP


def initialize_BioMIP(params, edge_type2decoder):
    return BioMIP(params, edge_type2decoder)


class BioMIP(nn.Module):
    def __init__(self, params, edge_type2decoder):
        super().__init__()
        self.params = params
        # Create two sets of intra-GNNs for small- and macro-molecules respectively
        self.small_intra_gnn = Intra_AttentiveFP(
            node_feat_size=params.atom_insize,
            edge_feat_size=params.bond_insize,
            graph_feat_size=params.intra_out_dim
        )
        # self.macro_intra_gnn = Intra_GAT()
        self.macro_intra_gnn = Intra_AttentiveFP(
            node_feat_size=params.aa_node_insize,
            edge_feat_size=params.aa_edge_insize,
            graph_feat_size=params.intra_out_dim
        )
        self.init_inter_with_intra = True  # params.init_inter_with_intra
        # Create a stack of inter-GNNs
        # self.inter_gnn = InterView_RGCN(params)
        self.inter_gnn = RGCN(params.inp_dim,
                              params.inp_dim,
                              params.emb_dim,
                              rel_names=params.rel2id
                              )
        self.edge_type2decoder = {}
        # for etype, predictor in edge_type2decoder:
        #     self.edge_type2decoder[etype] =

    def forward(self, mol_structs, pos_graph, neg_graph, pred_rels:list):
        small_bg = dgl.batch(mol_structs['small'])  # to("cuda:x")
        small_mol_feats = self.small_intra_gnn(small_bg, small_bg.ndata['nfeats'].to(torch.float32),
                                               small_bg.edata['efeats'].to(torch.float32))
        target_bg = dgl.batch(mol_structs['target'])  # .to("cuda:0")
        target_mol_feats = self.macro_intra_gnn(target_bg, target_bg.ndata['nfeats'].to(torch.float32),
                                                target_bg.edata['efeats'].to(torch.float32))
        if 'bio' in mol_structs:
            bio_bg = dgl.batch(mol_structs['bio'])
            bio_mol_feats = self.small_intra_gnn(bio_bg, bio_bg.ndata['nfeats'].to(torch.float32),
                                                 bio_bg.edata['efeats'].to(torch.float32))
        for g in [pos_graph, neg_graph]:
            g.nodes['drug'].data['intra'] = torch.cat([small_mol_feats, bio_mol_feats], 0) \
                if 'bio' in mol_structs else small_mol_feats
            g.nodes['target'].data['intra'] = target_mol_feats
            g.ndata['repr'] = self.inter_gnn(g)
        print(pos_graph.ndata['repr'], neg_graph.ndata['repr'])

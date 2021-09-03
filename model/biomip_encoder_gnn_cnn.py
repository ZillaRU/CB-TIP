import dgl
import torch
import torch.nn as nn

from model.inter_encoder.dgl_rgcn import RGCN
from model.intra_encoder.intra_GNN import Intra_AttentiveFP


class BioMIP_encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Create two sets of intra-GNNs for small- and macro-molecules respectively
        self.small_intra = Intra_AttentiveFP(
            node_feat_size=params.atom_insize,
            edge_feat_size=params.bond_insize,
            graph_feat_size=params.intra_out_dim
        )
        self.macro_intra = nn.ModuleDict({
            'embedding': nn.Embedding(num_embeddings=num_macro, embedding_dim=32),
            'conv': nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8),
            'fcl': nn.Linear(32 * 121, output_dim)
        })

        self.init_inter_with_intra = True  # params.init_inter_with_intra
        # Create a stack of inter-GNNs
        self.inter_gnn = RGCN(params.inp_dim,
                              params.inp_dim,
                              params.emb_dim,
                              rel_names=params.rel2id
                              )

    def forward(self, mol_structs, pos_graph):
        small_bg = dgl.batch(mol_structs['small'])  # to("cuda:x")
        small_mol_feats = self.small_intra(small_bg, small_bg.ndata['nfeats'].to(torch.float32),
                                           small_bg.edata['efeats'].to(torch.float32))
        target_bg = dgl.batch(mol_structs['target'])  # .to("cuda:0")
        target_mol_feats = self.macro_intra(target_bg, target_bg.ndata['nfeats'].to(torch.float32),
                                            target_bg.edata['efeats'].to(torch.float32))
        bio_mol_feats = None
        if 'bio' in mol_structs:
            bio_bg = dgl.batch(mol_structs['bio'])
            bio_mol_feats = self.small_intra(bio_bg, bio_bg.ndata['nfeats'].to(torch.float32),
                                             bio_bg.edata['efeats'].to(torch.float32))
        pos_graph.nodes['drug'].data['intra'] = torch.cat([small_mol_feats, bio_mol_feats], 0) \
            if 'bio' in mol_structs else small_mol_feats
        pos_graph.nodes['target'].data['intra'] = target_mol_feats
        return (small_mol_feats, bio_mol_feats, target_mol_feats), pos_graph.self.inter_gnn(g)

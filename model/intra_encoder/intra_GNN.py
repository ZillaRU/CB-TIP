import torch.nn as nn


from model.intra_encoder import AttentiveFPGNN
from model.intra_readout import AttentiveFPReadout


class Intra_AttentiveFP(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(Intra_AttentiveFP, self).__init__()
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size)
        # self.predict = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(graph_feat_size, n_tasks)
        # )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.init_context.reset_parameters()
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_feat_size)
            Input edge features. E for the number of edges.

        Returns
        -------
        node_feats : float32 tensor of shape (V, graph_feat_size)
            Updated node representations.
        """
        node_feats = self.gnn(g, node_feats, edge_feats)
        g_feats = self.readout(g, node_feats)
        return g_feats


class Intra_GAT(nn.Module):
    pass


class Intra_GCN(nn.Module):
    pass

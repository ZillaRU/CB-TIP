import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class RGCNLayer(nn.Module):
    def __init__(self, device,
                 inp_dim, out_dim,
                 aggregator,
                 bias=None,
                 activation=None,
                 dropout=0.0, edge_dropout=0.0,
                 is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, attn_rel_emb=None):
        # print("g, attn_rel_emb", g)
        self.propagate(g, attn_rel_emb)
        # apply bias and activation
        node_repr = {}
        for ntype in g.ntypes:
            # print("---", ntype, g.nodes[ntype].data)
            node_repr[ntype] = g.nodes[ntype].data['h']
        # print("g.ndata['h']", node_repr)
        for ntype in g.ntypes:
            if self.bias:
                node_repr[ntype] = node_repr[ntype] + self.bias
            if self.activation:
                # node_repr = self.activation(node_repr)
                node_repr[ntype] = self.activation(node_repr[ntype])
            if self.dropout:
                # node_repr = self.dropout(node_repr)
                node_repr[ntype] = self.dropout(node_repr[ntype])

            g.nodes[ntype].data['h'] = node_repr[ntype]
        # print("g.ndata['h']", g.ndata['h'])
        # g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        for ntype in g.ntypes:
            if self.is_input_layer:
                g.nodes[ntype].data['repr'] = g.nodes[ntype].data['h']
                # print("g.nodes[ntype].data['repr'].shape:", g.nodes[ntype].data['h'].shape)
                # print("g.nodes[ntype].data['repr']", g.nodes[ntype].data['repr'])
            else:
                # print(g.nodes[ntype].data['repr'].shape, g.nodes[ntype].data['h'].shape)
                g.nodes[ntype].data['repr'] = torch.cat(
                    [g.nodes[ntype].data['repr'], g.nodes[ntype].data['h']]
                    , dim=1
                )
                # g.ndata['repr'][ntype] = torch.cat([g.ndata['repr'][ntype], g.ndata['h'][ntype].unsqueeze(1)], dim=1)
                # print("g.nodes[ntype].data['repr']", g.nodes[ntype].data['repr'].shape)


class RGCNBasisLayer(RGCNLayer):
    def __init__(self,
                 inp_dim, out_dim,
                 aggregator,
                 attn_rel_emb_dim,
                 num_rels,
                 rel2id,
                 device,
                 num_bases=-1,
                 bias=None,
                 activation=None,
                 dropout=0.0, edge_dropout=0.0,
                 is_input_layer=False, has_attn=False, ):
        super(
            RGCNBasisLayer, self).__init__(
            device,
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.rel2id = rel2id
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn
        self.device = device

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim)).to(device)
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases)).to(device)

        if self.has_attn:
            # self.A = nn.Linear(2 * self.inp_dim + 2 * self.attn_rel_emb_dim, inp_dim)
            self.A = nn.Linear(2 * self.inp_dim + self.attn_rel_emb_dim, inp_dim).to(device)
            self.B = nn.Linear(inp_dim, 1).to(device)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim)).to(device)

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, attn_rel_emb=None):
        # generate all weights from bases
        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)
        # print("weight.device: ", weight.device)
        for etype in g.canonical_etypes:
            g.edges[etype].data['w'] = self.edge_dropout(torch.ones(g.number_of_edges(etype), 1).to(weight.device))
            # g.edges[etype].data['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))
            # print(f"g.edges[{etype}].data['w']", g.edges[etype].data['w'])

        # g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))
        # input_ = 'intra' if self.is_input_layer else 'h'
        input_ = 'h'

        def msg_func(edges):
            etype_id = self.rel2id[edges.canonical_etype[1]]
            # print("edges.rel", etype_id)
            # print("['h']", edges.src['h'], edges.dst['h'])  # '_ID' 'intra' 'h'
            # print("edges.data", edges.data)
            # w = weight.index_select(0, edges.data['type'])
            w = weight.index_select(0, torch.LongTensor([etype_id] * edges.batch_size()).to(self.device))
            msg = edges.data['w'] * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (B, F)

            if self.has_attn:
                # e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(edges.data['type']),
                #                attn_rel_emb(edges.data['label'])], dim=1)
                e = torch.cat([edges.src[input_], edges.dst[input_], attn_rel_emb(etype_id)], dim=1)
                a = torch.sigmoid(self.B(F.relu(self.A(e))))
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        # g.multi_update_all({
        #     : msg_func
        # }}
        # , self.aggregator, None)
        for etype in g.canonical_etypes:
            # print(src_type, etype, des_type)
            # print("canonical_etypes ['h']\n", g.nodes['drug'].data, g.nodes['target'].data)  # '_ID' 'intra' 'h'
            g[etype].update_all(message_func=msg_func,
                                reduce_func=self.aggregator,
                                # apply_node_func=,
                                etype=etype)
            # print(" -- canonical_etypes ['h']\n", g.nodes['drug'].data, g.nodes['target'].data)  # '_ID' 'intra' 'h'

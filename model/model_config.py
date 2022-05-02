from model import GNN_GNN_GNN
from model import FF_MI_max
from model import MultiInnerProductDecoder


def initialize_BioMIP(params):
    return GNN_GNN_GNN(params), \
           MultiInnerProductDecoder(params.inp_dim, params.num_rels, params).to(params.device), \
           MultiInnerProductDecoder(params.inp_dim, params.num_rels, params).to(params.device), \
           FF_MI_max(params).to(params.device)
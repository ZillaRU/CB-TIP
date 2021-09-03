from biomip_encoder_gnn_cnn import BioMIP_encoder as GNN_CNN_GNN
# from biomip_encoder_gnn_rnn import BioMIP_encoder as GNN_RNN_GNN
from biomip_encoder_gnn_gnn import BioMIP_encoder as GNN_GNN_GNN

from biomip_decoder import MultiInnerProductDecoder, DEDICOMDecoder, DistMultDecoder, BilinearDecoder, NNDecoder
from model_info import GcnInfomax1


def initialize_BioMIP(params):
    return GNN_GNN_GNN(params), MultiInnerProductDecoder(params.inp_dim, params.num_rels), GcnInfomax1(params)

# Define a Heterograph Conv model
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph):
        # 输入是节点的特征字典
        # print("# 输入是节点的特征字典", {
        #     'drug': graph.nodes['drug'].data['intra'],
        #     'target': graph.nodes['target'].data['intra']
        # })
        # print(graph)
        h = self.conv1(graph, {
            'drug': graph.nodes['drug'].data['intra'],
            'target': graph.nodes['target'].data['intra']
        })

        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

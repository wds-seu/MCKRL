import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl.function as fn
from model import Attention

class MCKRL(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super(MC_RGCN, self).__init__()
        self.sage0 = RGCN(in_features, hidden_features, out_features, rel_names[0])
        self.sage1 = RGCN(in_features, hidden_features, out_features, rel_names[1])
        self.sage2 = RGCN(in_features, hidden_features, out_features, rel_names[2])
        self.sage3 = RGCN(in_features, hidden_features, out_features, rel_names[3])
        self.attention = Attention(out_features)

    def forward(self, g_list, x):
        h_list = []

        for i in range(len(g_list)):
            h_list.append(x)

        h_list[0] = self.sage0(g_list[0], x)
        h_list[1] = self.sage1(g_list[1], x)
        h_list[2] = self.sage2(g_list[2], x)
        h_list[3] = self.sage3(g_list[3], x)

        # emb = torch.stack([h_list[0], h_list[1], h_list[2], h_list[3]], dim=1)
        # emb = self.attention(emb)

        # d1 = h_list[0]['drug'] + h_list[3]['drug']
        # d1 = d1 / 2
        #
        # p1 = h_list[0]['protein'] + h_list[3]['protein']
        # p1 = p1 / 2
        # emb_drug = d1
        # emb_pro = p1
        # emb_drug = (h_list[0]['drug'] + h_list[1]['drug'] + h_list[2]['drug'] + h_list[3]['drug']) / 4
        # emb_pro = (h_list[0]['protein'] + h_list[1]['protein'] + h_list[2]['protein'] + h_list[3]['protein']) / 4

        # emb_drug = torch.stack([h_list[3]['drug']], dim=1)
        # emb_pro = torch.stack([h_list[3]['protein']], dim=1)
        emb_drug = torch.stack([h_list[0]['drug'], h_list[2]['drug'], h_list[3]['drug']], dim=1)
        emb_pro = torch.stack([h_list[0]['protein'], h_list[2]['protein'], h_list[3]['protein']], dim=1)

        # emb_drug = torch.stack([h_list[0]['drug']], dim=1)
        # emb_pro = torch.stack([h_list[0]['protein']], dim=1)

        beta_drug, emb_drug = self.attention(emb_drug)
        beta_pro,emb_pro = self.attention(emb_pro)
        emb = {'drug': emb_drug, 'protein': emb_pro}
        return emb, beta_drug, beta_pro


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

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        return h

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']
import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
from vocab import Vocab
import numpy as np
import os
import scipy.sparse as sp
import random


drug_drug_csv = pd.read_csv('./dataset/db_up/drugs2423_interactions.csv', delimiter=',',
                            names=['drug1', 'drug2'])
drug1_np = drug_drug_csv['drug1'].values
drug2_np = drug_drug_csv['drug2'].values

drug_pro_csv = pd.read_csv('./dataset/db_up/drugs1861_target_protein1950.csv', delimiter=',',
                           names=['drug', 'action', 'pro'])
drug_np = drug_pro_csv['drug'].values
pro_np = drug_pro_csv['pro'].values

pro_pro_csv = pd.read_csv('./dataset/db_up/proteins7037_interactions.csv', delimiter=',',
                          names=['pro1', 'pro2'])
pro1_np = pro_pro_csv['pro1'].values
pro2_np = pro_pro_csv['pro2'].values

drugs2423_csv = pd.read_csv('./dataset/db_up/drugs2423.csv', delimiter=',', names=['drug'])
drugs2423 = drugs2423_csv['drug'].values
proteins7954_csv = pd.read_csv('./dataset/db_up/proteins7954.csv', delimiter=',', names=['pro'])
proteins7954 = proteins7954_csv['pro'].values

drug_np_unique = np.unique(drug_np)
for i in drugs2423:
    if i not in np.unique(drug_np_unique):
        print(i)


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class DBDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='drugbank_uniprot')

    def process(self):
        # set random seed
        set_random_seed()

        # ratio
        validation_ratio = 0.4
        test_ratio = 0.4

        # nodes to ids
        drugs2423_vocab = Vocab(drugs2423.tolist())

        drug_ids = []
        drug_ids_np = np.array(drugs2423_vocab.nodes2ids(drugs2423))
        drug_ids.append([drug_ids_np, drugs2423])
        drug_ids = np.array(drug_ids, dtype=np.str).reshape(2, 2423)
        drug_ids = np.transpose(drug_ids)
        # print(drug_ids)
        np.savetxt("./dataset/db_up/drug_ids.csv", drug_ids, fmt="%s", delimiter=",",
                   encoding="utf-8")

        proteins7954_vocab = Vocab(proteins7954.tolist())

        # drug-drug interaction
        dd_d1 = drugs2423_vocab.nodes2ids(drug1_np)

        drug_neighbor_num = []
        drugs_tensor = torch.from_numpy(np.array(dd_d1)).float()
        drugs_unique_values = torch.unique(drugs_tensor)
        counts = torch.histc(drugs_tensor, bins=len(drugs_unique_values), min=0, max=len(drugs_unique_values) - 1).long()
        drug_neighbor_num.append([np.array(drugs_unique_values), np.array(counts)])
        drug_neighbor_num = np.array(drug_neighbor_num).reshape(2, 2423)
        drug_neighbor_num = np.transpose(drug_neighbor_num)
        np.savetxt("./dataset/db_up/drug_neighbor_num.csv", drug_neighbor_num, fmt="%s", delimiter=",",
                   encoding="utf-8")

        dd_d2 = drugs2423_vocab.nodes2ids(drug2_np)
        # drug-protein interaction
        dp_d = drugs2423_vocab.nodes2ids(drug_np)
        dp_p = proteins7954_vocab.nodes2ids(pro_np)
        # protein-protein interaction
        pp_p1 = proteins7954_vocab.nodes2ids(pro1_np)
        pp_p2 = proteins7954_vocab.nodes2ids(pro2_np)
        # read .content
        idx_feartures = np.genfromtxt("./dataset/db_up/entities.content", dtype=np.dtype(str))
        d_features = np.array(idx_feartures[:len(drugs2423), 1:], dtype=np.float32)
        p_features = np.array(idx_feartures[len(drugs2423):, 1:], dtype=np.float32)
        d_node_features = torch.from_numpy(d_features)
        p_node_features = torch.from_numpy(p_features)

        """
        homo graph: drug drug interaction
        """
        # construct homo graph(dd)
        dd_src = torch.from_numpy(np.array(dd_d1))
        dd_dst = torch.from_numpy(np.array(dd_d2))
        g = dgl.graph((dd_src, dd_dst), num_nodes=drugs2423.shape[0])
        g.ndata['feat'] = d_node_features

        u, v = g.edges()
        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * validation_ratio)
        val_size = int(len(eids) * test_ratio) + test_size
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        val_pos_u, val_pos_v = u[eids[test_size: val_size]], v[eids[test_size: val_size]]
        train_pos_u, train_pos_v = u[eids[val_size:]], v[eids[val_size:]]


        # Find all negative edges and split them for training and testing
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(drugs2423.shape[0], drugs2423.shape[0]))
        adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
        val_neg_u, val_neg_v = neg_u[neg_eids[test_size: val_size]], neg_v[neg_eids[test_size: val_size]]
        train_neg_u, train_neg_v = neg_u[neg_eids[val_size:]], neg_v[neg_eids[val_size:]]

        self.train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
        self.train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())
        self.val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=g.number_of_nodes())
        self.val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=g.number_of_nodes())
        self.test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
        self.test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())
        self.train_g = dgl.remove_edges(g, eids[:val_size])

        """
        entire hetero graph
        """
        entire_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (dd_d1, dd_d2),
            ('drug', 'has_target', 'protein'): (dp_d, dp_p),
            ('protein', 'is_target_of', 'drug'): (dp_p, dp_d),
            ('protein', 'pp_int', 'protein'): (pp_p1, pp_p2)
        }
        entire_num_nodes_dict = {'drug': drugs2423.shape[0], 'protein': proteins7954.shape[0]}
        self.entire_hete_g = dgl.heterograph(entire_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)

        # count num of nodes
        self.num_of_drugs = self.entire_hete_g.num_nodes('drug')
        self.num_of_proteins = self.entire_hete_g.num_nodes('protein')
        self.num_of_dd = self.entire_hete_g.num_edges('dd_int')
        self.num_of_dp = self.entire_hete_g.num_edges('has_target')
        self.num_of_pd = self.entire_hete_g.num_edges('is_target_of')
        self.num_of_pp = self.entire_hete_g.num_edges('pp_int')

        self.entire_hete_g.nodes['drug'].data['feat'] = d_node_features
        self.entire_hete_g.nodes['protein'].data['feat'] = p_node_features

        # divide  for train,val,test ----- pos
        # eids
        dp_u, dp_v = self.entire_hete_g.edges(etype='has_target')
        pd_u, pd_v = self.entire_hete_g.edges(etype='is_target_of')
        pp_u, pp_v = self.entire_hete_g.edges(etype='pp_int')

        dp_eids = np.arange(self.num_of_dp)
        dp_eids = np.random.permutation(dp_eids)
        pd_eids = np.arange(self.num_of_pd)
        pd_eids = np.random.permutation(pd_eids)
        pp_eids = np.arange(self.num_of_pp)
        pp_eids = np.random.permutation(pp_eids)

        # test size
        # dd_test_size = test_size
        dp_test_size = int(len(dp_eids) * test_ratio)
        pd_test_size = int(len(pd_eids) * test_ratio)
        pp_test_size = int(len(pp_eids) * test_ratio)

        # val size
        # dd_val_size = val_size
        dp_val_size = int(len(dp_eids) * validation_ratio) + dp_test_size
        pd_val_size = int(len(pd_eids) * validation_ratio) + pd_test_size
        pp_val_size = int(len(pp_eids) * validation_ratio) + pp_test_size

        # test pos edges
        dp_test_pos_u, dp_test_pos_v = dp_u[dp_eids[:dp_test_size]], dp_v[dp_eids[:dp_test_size]]
        pd_test_pos_u, pd_test_pos_v = pd_u[pd_eids[:pd_test_size]], pd_v[pd_eids[:pd_test_size]]
        pp_test_pos_u, pp_test_pos_v = pp_u[pp_eids[:pp_test_size]], pp_v[pp_eids[:pp_test_size]]

        """hete: construct test pos graph"""
        test_pos_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (test_pos_u, test_pos_v),
            ('drug', 'has_target', 'protein'): (dp_test_pos_u, dp_test_pos_v),
            ('protein', 'is_target_of', 'drug'): (pd_test_pos_u, pd_test_pos_v),
            ('protein', 'pp_int', 'protein'): (pp_test_pos_u, pp_test_pos_v)
        }

        self.hete_test_pos_g = dgl.heterograph(test_pos_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)
        self.hete_test_pos_g.nodes['drug'].data['feat'] = d_node_features
        self.hete_test_pos_g.nodes['protein'].data['feat'] = p_node_features

        # val pos edges
        dp_val_pos_u, dp_val_pos_v = dp_u[dp_eids[dp_test_size: dp_val_size]], dp_v[dp_eids[dp_test_size: dp_val_size]]
        pd_val_pos_u, pd_val_pos_v = pd_u[pd_eids[pd_test_size: pd_val_size]], pd_v[pd_eids[pd_test_size: pd_val_size]]
        pp_val_pos_u, pp_val_pos_v = pp_u[pp_eids[pp_test_size: pp_val_size]], pp_v[pp_eids[pp_test_size: pp_val_size]] # 这里有bug


        """hete: construct val pos graph"""
        val_pos_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (val_pos_u, val_pos_v),
            ('drug', 'has_target', 'protein'): (dp_val_pos_u, dp_val_pos_v),
            ('protein', 'is_target_of', 'drug'): (pd_val_pos_u, pd_val_pos_v),
            ('protein', 'pp_int', 'protein'): (pp_val_pos_u, pp_val_pos_v)
        }

        self.hete_val_pos_g = dgl.heterograph(val_pos_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)
        self.hete_val_pos_g.nodes['drug'].data['feat'] = d_node_features
        self.hete_val_pos_g.nodes['protein'].data['feat'] = p_node_features

        # train pos edges
        dp_train_pos_u, dp_train_pos_v = dp_u[dp_eids[dp_val_size:]], dp_v[dp_eids[dp_val_size:]]
        pd_train_pos_u, pd_train_pos_v = pd_u[pd_eids[pd_val_size:]], pd_v[pd_eids[pd_val_size:]]
        pp_train_pos_u, pp_train_pos_v = pp_u[pp_eids[pp_val_size:]], pp_v[pp_eids[pp_val_size:]]

        """hete: construct train pos graph"""
        train_pos_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (train_pos_u, train_pos_v),
            ('drug', 'has_target', 'protein'): (dp_train_pos_u, dp_train_pos_v),
            ('protein', 'is_target_of', 'drug'): (pd_train_pos_u, pd_train_pos_v),
            ('protein', 'pp_int', 'protein'): (pp_train_pos_u, pp_train_pos_v)
        }

        self.hete_train_pos_g = dgl.heterograph(train_pos_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)
        self.hete_train_pos_g.nodes['drug'].data['feat'] = d_node_features
        self.hete_train_pos_g.nodes['protein'].data['feat'] = p_node_features

        # divide  for train,val,test ----- neg
        # dp
        dp_adj = sp.coo_matrix((np.ones(len(dp_u)), (dp_u.numpy(), dp_v.numpy())),
                               shape=(drugs2423.shape[0], proteins7954.shape[0]))
        dp_adj_neg = 1 - dp_adj.todense() - np.eye(N=drugs2423.shape[0], M=proteins7954.shape[0])
        dp_neg_u, dp_neg_v = np.where(dp_adj_neg != 0)

        dp_neg_eids = np.random.choice(len(dp_neg_u), self.num_of_dp)
        dp_test_neg_u, dp_test_neg_v = dp_neg_u[dp_neg_eids[:dp_test_size]], \
                                       dp_neg_v[dp_neg_eids[:dp_test_size]]
        dp_val_neg_u, dp_val_neg_v = dp_neg_u[dp_neg_eids[dp_test_size: dp_val_size]], \
                                     dp_neg_v[dp_neg_eids[dp_test_size: dp_val_size]]
        dp_train_neg_u, dp_train_neg_v = dp_neg_u[dp_neg_eids[dp_val_size:]], dp_neg_v[dp_neg_eids[dp_val_size:]]

        # pd
        pd_adj = sp.coo_matrix((np.ones(len(pd_u)), (pd_u.numpy(), pd_v.numpy())),
                               shape=(proteins7954.shape[0], drugs2423.shape[0]))
        pd_adj_neg = 1 - pd_adj.todense() - np.eye(N=proteins7954.shape[0], M=drugs2423.shape[0])
        pd_neg_u, pd_neg_v = np.where(pd_adj_neg != 0)

        pd_neg_eids = np.random.choice(len(pd_neg_u), self.num_of_pd)
        pd_test_neg_u, pd_test_neg_v = pd_neg_u[pd_neg_eids[:pd_test_size]], \
                                       pd_neg_v[pd_neg_eids[:pd_test_size]]
        pd_val_neg_u, pd_val_neg_v = pd_neg_u[pd_neg_eids[pd_test_size: pd_val_size]], \
                                     pd_neg_v[pd_neg_eids[pd_test_size:pd_val_size]]
        pd_train_neg_u, pd_train_neg_v = pd_neg_u[pd_neg_eids[pd_val_size:]], pd_neg_v[pd_neg_eids[pd_val_size:]]

        # pp
        pp_adj = sp.coo_matrix((np.ones(len(pp_u)), (pp_u.numpy(), pp_v.numpy())),
                               shape=(proteins7954.shape[0], proteins7954.shape[0]))
        pp_adj_neg = 1 - pp_adj.todense() - np.eye(N=proteins7954.shape[0], M=proteins7954.shape[0])
        pp_neg_u, pp_neg_v = np.where(pp_adj_neg != 0)

        pp_neg_eids = np.random.choice(len(pp_neg_u), self.num_of_pp)
        pp_test_neg_u, pp_test_neg_v = pp_neg_u[pp_neg_eids[:pp_test_size]], \
                                       pp_neg_v[pp_neg_eids[:pp_test_size]]
        pp_val_neg_u, pp_val_neg_v = pp_neg_u[pp_neg_eids[pp_test_size: pp_val_size]], \
                                     pp_neg_v[pp_neg_eids[pp_test_size: pp_val_size]]
        pp_train_neg_u, pp_train_neg_v = pp_neg_u[pp_neg_eids[pp_val_size:]], pp_neg_v[pp_neg_eids[pp_val_size:]]

        """hete: construct test neg graph"""
        test_neg_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (test_neg_u, test_neg_v),
            ('drug', 'has_target', 'protein'): (dp_test_neg_u, dp_test_neg_v),
            ('protein', 'is_target_of', 'drug'): (pd_test_neg_u, pd_test_neg_v),
            ('protein', 'pp_int', 'protein'): (pp_test_neg_u, pp_test_neg_v)
        }

        self.hete_test_neg_g = dgl.heterograph(test_neg_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)

        """hete: construct val neg graph"""
        val_neg_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (val_neg_u, val_neg_v),
            ('drug', 'has_target', 'protein'): (dp_val_neg_u, dp_val_neg_v),
            ('protein', 'is_target_of', 'drug'): (pd_val_neg_u, pd_val_neg_v),
            ('protein', 'pp_int', 'protein'): (pp_val_neg_u, pp_val_neg_v)
        }

        self.hete_val_neg_g = dgl.heterograph(val_neg_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)

        """hete: construct train neg graph"""
        train_neg_hetero_graph_dict = {
            ('drug', 'dd_int', 'drug'): (train_neg_u, train_neg_v),
            ('drug', 'has_target', 'protein'): (dp_train_neg_u, dp_train_neg_v),
            ('protein', 'is_target_of', 'drug'): (pd_train_neg_u, pd_train_neg_v),
            ('protein', 'pp_int', 'protein'): (pp_train_neg_u, pp_train_neg_v)
        }

        self.hete_train_neg_g = dgl.heterograph(train_neg_hetero_graph_dict, num_nodes_dict=entire_num_nodes_dict)

    def __getitem__(self, i):
        if i == 'hete_test_pos':     # hete_pos
            return self.hete_test_pos_g
        elif i == 'hete_val_pos':
            return self.hete_val_pos_g
        elif i == 'hete_train_pos':
            return self.hete_train_pos_g
        elif i == 'hete_test_neg':   # hete_neg
            return self.hete_test_neg_g
        elif i == 'hete_val_neg':
            return self.hete_val_neg_g
        elif i == 'hete_train_neg':
            return self.hete_train_neg_g
        elif i == 'hete_entire':
            return self.entire_hete_g
        elif i == 'homo_test_pos':    # homo_pos
            return self.test_pos_g
        elif i == 'homo_val_pos':
            return self.val_pos_g
        elif i == 'homo_train_pos':
            return self.train_pos_g
        elif i == 'homo_train_g':
            return self.train_g
        elif i == 'homo_test_neg':   # homo_neg
            return self.test_neg_g
        elif i == 'homo_val_neg':
            return self.val_neg_g
        elif i == 'homo_train_neg':
            return self.train_neg_g

    def __len__(self):
        return 11


dataset = DBDataset()
dgl.save_graphs('D:/pyProjects/dataset/db_up/dg_up_pos.dgl', [dataset["hete_test_pos"], dataset["hete_val_pos"],
                                                  dataset["hete_train_pos"]])
dgl.save_graphs('D:/pyProjects/dataset/db_up/dg_up_neg.dgl', [dataset["hete_test_neg"], dataset["hete_val_neg"],
                                                  dataset["hete_train_neg"]])

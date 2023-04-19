import torch
import sklearn.metrics as metrics
import torch.nn.functional as F
import random
import numpy as np
import dgl


def compute_loss(pos_score, neg_score):
    """compute loss: Binary Cross Entropy
    """
    # scores = torch.cat([pos_score, neg_score])
    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    """compute auc: roc_auc_score in sklearn.metrics
    """
    # scores = torch.cat([pos_score, neg_score]).detach().numpy()
    scores = torch.sigmoid(torch.cat([pos_score, neg_score])).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return metrics.roc_auc_score(labels, scores)


def compute_acc_f1(pos_score, neg_score):
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))
    # print(scores)
    pred = (scores > 0.5).float()
    pred = pred.numpy()
    labels = labels.numpy()
    pred = pred.reshape(labels.shape[0], 1)
    labels = labels.reshape(labels.shape[0], 1)
    correct = np.sum(labels == pred)
    acc = correct / labels.shape[0]
    print("correct: ", correct)
    print("labels: ", labels.shape[0])

    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))
    tn = np.sum((pred == 0) & (labels == 0))
    print("tp:{} | fp:{} | fn:{} | tn: {}", tp, fp, fn, tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return acc,f1


def compute_aupr(pos_score, neg_score):
    # scores = torch.cat([pos_score, neg_score]).detach().numpy()
    scores = torch.sigmoid(torch.cat([pos_score, neg_score])).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return metrics.average_precision_score(labels, scores)


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


def read_db_up():
    # (test_pos_g, val_pos_g, train_pos_g), _ = dgl.load_graphs('D:/pyProjects/dataset/db_up/dg_up_pos.dgl')
    # (test_neg_g, val_neg_g, train_neg_g), _ = dgl.load_graphs('D:/pyProjects/dataset/db_up/dg_up_neg.dgl')

    (test_pos_g, val_pos_g, train_pos_g), _ = dgl.load_graphs('D:/PycharmProjects/MakeDataset/decagon/biose_pos.dgl')
    (test_neg_g, val_neg_g, train_neg_g), _ = dgl.load_graphs('D:/PycharmProjects/MakeDataset/decagon/biose_neg.dgl')

    dg_up_pos = [test_pos_g, val_pos_g, train_pos_g]
    dg_up_neg = [test_neg_g, val_neg_g, train_neg_g]
    """打印 DrugBank + Uniprot 数据集相关信息"""
    print("# entities - ", end="")
    print("drugs: {}, proteins: {}".format(test_pos_g.num_nodes('drug'), test_pos_g.num_nodes('protein')))
    print("# edges - ", end="")
    for type in test_pos_g.etypes:
        print("{}: {}, ".format(type,
                                test_pos_g.num_edges(type) + val_pos_g.num_edges(type) + train_pos_g.num_edges(type))
              , end="")
    print("\n# tra edges - ", end="")
    for type in test_pos_g.etypes:
        print("{}: {}  ".format(type, train_pos_g.num_edges(type)), end="")
    print("\n# val edges - ", end="")
    for type in test_pos_g.etypes:
        print("{}: {}  ".format(type, val_pos_g.num_edges(type)), end="")
    print("\n# tes edges - ", end="")
    for type in test_pos_g.etypes:
        print("{}: {}  ".format(type, test_pos_g.num_edges(type)), end="")
    print()

    '''drug-drug interaction'''
    etypes = train_pos_g.canonical_etypes
    dd_train_pos_g = dgl.edge_type_subgraph(train_pos_g, [etypes[0]])
    dd_train_neg_g = dgl.edge_type_subgraph(train_neg_g, [etypes[0]])
    dd_val_pos_g = dgl.edge_type_subgraph(val_pos_g, [etypes[0]])
    dd_val_neg_g = dgl.edge_type_subgraph(val_neg_g, [etypes[0]])
    dd_test_pos_g = dgl.edge_type_subgraph(test_pos_g, [etypes[0]])
    dd_test_neg_g = dgl.edge_type_subgraph(test_neg_g, [etypes[0]])

    '''不同图关系'''

    '''g1: d-d, d-p'''
    train_pos_g1 = dgl.edge_type_subgraph(train_pos_g, [etypes[0], etypes[1]])
    train_pos_g1 = dgl.to_homogeneous(train_pos_g1)
    train_neg_g1 = dgl.edge_type_subgraph(train_neg_g, [etypes[0], etypes[1]])
    train_neg_g1 = dgl.to_homogeneous(train_neg_g1)

    val_pos_g1 = dgl.edge_type_subgraph(val_pos_g, [etypes[0], etypes[1]])
    val_pos_g1 = dgl.to_homogeneous(val_pos_g1)
    val_neg_g1 = dgl.edge_type_subgraph(val_neg_g, [etypes[0], etypes[1]])
    val_neg_g1 = dgl.to_homogeneous(val_neg_g1)

    test_pos_g1 = dgl.edge_type_subgraph(test_pos_g, [etypes[0], etypes[1]])
    test_pos_g1 = dgl.to_homogeneous(test_pos_g1)
    test_neg_g1 = dgl.edge_type_subgraph(test_neg_g, [etypes[0], etypes[1]])
    test_neg_g1 = dgl.to_homogeneous(test_neg_g1)

    '''g2: p-d, p-p'''
    train_pos_g2 = dgl.edge_type_subgraph(train_pos_g, [etypes[2], etypes[3]])
    train_pos_g2 = dgl.to_homogeneous(train_pos_g2)
    train_neg_g2 = dgl.edge_type_subgraph(train_neg_g, [etypes[2], etypes[3]])
    train_neg_g2 = dgl.to_homogeneous(train_neg_g2)

    val_pos_g2 = dgl.edge_type_subgraph(val_pos_g, [etypes[2], etypes[3]])
    val_pos_g2 = dgl.to_homogeneous(val_pos_g2)
    val_neg_g2 = dgl.edge_type_subgraph(val_neg_g, [etypes[2], etypes[3]])
    val_neg_g2 = dgl.to_homogeneous(val_neg_g2)

    test_pos_g2 = dgl.edge_type_subgraph(test_pos_g, [etypes[2], etypes[3]])
    test_pos_g2 = dgl.to_homogeneous(test_pos_g2)
    test_neg_g2 = dgl.edge_type_subgraph(test_neg_g, [etypes[2], etypes[3]])
    test_neg_g2 = dgl.to_homogeneous(test_neg_g2)

    '''g_inter: d-p, p-d'''
    train_pos_g_inter = dgl.edge_type_subgraph(train_pos_g, [etypes[1], etypes[2]])
    train_pos_g_inter = dgl.to_homogeneous(train_pos_g_inter)
    train_neg_g_inter = dgl.edge_type_subgraph(train_neg_g, [etypes[1], etypes[2]])
    train_neg_g_inter = dgl.to_homogeneous(train_neg_g_inter)

    val_pos_g_inter = dgl.edge_type_subgraph(val_pos_g, [etypes[1], etypes[2]])
    val_pos_g_inter = dgl.to_homogeneous(val_pos_g_inter)
    val_neg_g_inter = dgl.edge_type_subgraph(val_neg_g, [etypes[1], etypes[2]])
    val_neg_g_inter = dgl.to_homogeneous(val_neg_g_inter)

    test_pos_g_inter = dgl.edge_type_subgraph(test_pos_g, [etypes[1], etypes[2]])
    test_pos_g_inter = dgl.to_homogeneous(test_pos_g_inter)
    test_neg_g_inter = dgl.edge_type_subgraph(test_neg_g, [etypes[1], etypes[2]])
    test_neg_g_inter = dgl.to_homogeneous(test_neg_g_inter)

    '''g_sum: d-d, d-p'''
    train_pos_g_sum = train_pos_g
    train_pos_g_sum = dgl.to_homogeneous(train_pos_g_sum)
    train_neg_g_sum = train_neg_g
    train_neg_g_sum = dgl.to_homogeneous(train_neg_g_sum)

    val_pos_g_sum = val_pos_g
    val_pos_g_sum = dgl.to_homogeneous(val_pos_g_sum)
    val_neg_g_sum = val_neg_g
    val_neg_g_sum = dgl.to_homogeneous(val_neg_g_sum)

    test_pos_g_sum = test_pos_g
    test_pos_g_sum = dgl.to_homogeneous(test_pos_g_sum)
    test_neg_g_sum = test_neg_g
    test_neg_g_sum = dgl.to_homogeneous(test_neg_g_sum)

    g_train_pos_list = [train_pos_g1, train_pos_g2, train_pos_g_inter, train_pos_g_sum]
    g_train_neg_list = [train_neg_g1, train_neg_g2, train_neg_g_inter, train_neg_g_sum]
    g_val_pos_list = [val_pos_g1, val_pos_g2, val_pos_g_inter, val_pos_g_sum]
    g_val_neg_list = [val_neg_g1, val_neg_g2, val_neg_g_inter, val_neg_g_sum]
    g_test_pos_list = [test_pos_g1, test_pos_g2, test_pos_g_inter, test_pos_g_sum]
    g_test_neg_list = [test_neg_g1, test_neg_g2, test_neg_g_inter, test_neg_g_sum]
    dd_pos_list = [dd_train_pos_g, dd_val_pos_g, dd_test_pos_g]
    dd_neg_list = [dd_train_neg_g, dd_val_neg_g, dd_test_neg_g]

    return g_train_pos_list, g_train_neg_list, g_val_pos_list, g_val_neg_list, g_test_pos_list, g_test_neg_list, \
           dg_up_pos, dg_up_neg, dd_pos_list, dd_neg_list

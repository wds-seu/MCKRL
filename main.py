import argparse


from early_stopping import EarlyStopping
from model import *
from utils import *
from predictor import *


set_random_seed()
pred = HeteroDotProductPredictor()

def evalute_xx(model, h, g, val_pos_g, val_neg_g, e_list):
    model.eval()
    with torch.no_grad():
        val_loss = compute_loss_xx(val_pos_score_0, val_neg_score_0)
        val_auc = compute_roc_auc_xx(val_pos_score_0, val_neg_score_0)
        return val_loss, val_auc

def compute_loss_xx(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    scores = torch.sigmoid(torch.cat([pos_score, neg_score]))
    return (1 - scores[:pos_score.shape[0]] + scores[pos_score.shape[0]:].view(n_edges, -1)).clamp(min=0).mean()


def compute_roc_auc_xx(pos_score, neg_score):
    scores = torch.sigmoid(torch.cat([pos_score, neg_score])).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return metrics.roc_auc_score(labels, scores)


def main(args):
    (test_pos_g, val_pos_g, train_pos_g), _ = dgl.load_graphs('D:/pyProjects/dataset/db_up/dg_up_pos.dgl')
    (test_neg_g, val_neg_g, train_neg_g), _ = dgl.load_graphs('D:/pyProjects/dataset/db_up/dg_up_neg.dgl')

    dg_up_pos = [test_pos_g, val_pos_g, train_pos_g]
    dg_up_neg = [test_neg_g, val_neg_g, train_neg_g]
    etypes = train_pos_g.canonical_etypes

    '''drug-drug interaction'''
    etypes = train_pos_g.canonical_etypes
    dd_train_pos_g = dgl.edge_type_subgraph(train_pos_g, [etypes[0]])
    dd_train_neg_g = dgl.edge_type_subgraph(train_neg_g, [etypes[0]])
    dd_val_pos_g = dgl.edge_type_subgraph(val_pos_g, [etypes[0]])
    dd_val_neg_g = dgl.edge_type_subgraph(val_neg_g, [etypes[0]])
    dd_test_pos_g = dgl.edge_type_subgraph(test_pos_g, [etypes[0]])
    dd_test_neg_g = dgl.edge_type_subgraph(test_neg_g, [etypes[0]])

    '''g1: d-d, d-p'''
    train_pos_g1 = dgl.edge_type_subgraph(train_pos_g, [etypes[0], etypes[1]])
    train_neg_g1 = dgl.edge_type_subgraph(train_neg_g, [etypes[0], etypes[1]])

    val_pos_g1 = dgl.edge_type_subgraph(val_pos_g, [etypes[0], etypes[1]])
    val_neg_g1 = dgl.edge_type_subgraph(val_neg_g, [etypes[0], etypes[1]])

    test_pos_g1 = dgl.edge_type_subgraph(test_pos_g, [etypes[0], etypes[1]])
    test_neg_g1 = dgl.edge_type_subgraph(test_neg_g, [etypes[0], etypes[1]])

    '''g2: p-d, p-p'''
    train_pos_g2 = dgl.edge_type_subgraph(train_pos_g, [etypes[2], etypes[3]])
    train_neg_g2 = dgl.edge_type_subgraph(train_neg_g, [etypes[2], etypes[3]])

    val_pos_g2 = dgl.edge_type_subgraph(val_pos_g, [etypes[2], etypes[3]])
    val_neg_g2 = dgl.edge_type_subgraph(val_neg_g, [etypes[2], etypes[3]])

    test_pos_g2 = dgl.edge_type_subgraph(test_pos_g, [etypes[2], etypes[3]])
    test_neg_g2 = dgl.edge_type_subgraph(test_neg_g, [etypes[2], etypes[3]])

    '''g_inter: d-p, p-d'''
    train_pos_g_inter = dgl.edge_type_subgraph(train_pos_g, [etypes[1], etypes[2]])
    train_neg_g_inter = dgl.edge_type_subgraph(train_neg_g, [etypes[1], etypes[2]])

    val_pos_g_inter = dgl.edge_type_subgraph(val_pos_g, [etypes[1], etypes[2]])
    val_neg_g_inter = dgl.edge_type_subgraph(val_neg_g, [etypes[1], etypes[2]])

    test_pos_g_inter = dgl.edge_type_subgraph(test_pos_g, [etypes[1], etypes[2]])
    test_neg_g_inter = dgl.edge_type_subgraph(test_neg_g, [etypes[1], etypes[2]])

    '''g_sum: d-d, d-p'''
    train_pos_g_sum = train_pos_g
    train_neg_g_sum = train_neg_g

    val_pos_g_sum = val_pos_g
    val_neg_g_sum = val_neg_g

    test_pos_g_sum = test_pos_g
    test_neg_g_sum = test_neg_g

    # g_list
    train_g_pos_list = [train_pos_g1, train_pos_g2, train_pos_g_inter, train_pos_g_sum]
    val_g_pos_list = [val_pos_g1, val_pos_g2, val_pos_g_inter, val_pos_g_sum]
    test_g_pos_list = [test_pos_g1, test_pos_g2, test_pos_g_inter, test_pos_g_sum]

    # neg_list
    train_g_neg_list = [train_neg_g1, train_neg_g2, train_neg_g_inter, train_neg_g_sum]
    val_g_neg_list = [val_neg_g1, val_neg_g2, val_neg_g_inter, val_neg_g_sum]
    test_g_neg_list = [test_neg_g1, test_neg_g2, test_neg_g_inter, test_neg_g_sum]

    # etype
    g1_etype = [test_pos_g.etypes[0], test_pos_g.etypes[1]]
    g2_etype = [test_pos_g.etypes[2], test_pos_g.etypes[3]]
    g_inter_etype = [test_pos_g.etypes[1], test_pos_g.etypes[2]]
    g_sum_etype = [test_pos_g.etypes[0], test_pos_g.etypes[1], test_pos_g.etypes[2], test_pos_g.etypes[3]]
    etype_list = [g1_etype, g2_etype, g_inter_etype, g_sum_etype]


    # get feat
    train_pos_g = dg_up_pos[2]
    drug_features = train_pos_g.nodes['drug'].data['feat']
    protein_features = train_pos_g.nodes['protein'].data['feat']
    features = {'drug': drug_features, 'protein': protein_features}

    model = MCKRL(32, 64, 64, etype_list)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)

    e0 = train_pos_g.canonical_etypes[0]
    e1 = train_pos_g.canonical_etypes[1]
    e2 = train_pos_g.canonical_etypes[2]
    e3 = train_pos_g.canonical_etypes[3]
    e_list = [e0, e1, e2, e3]
    print(e_list)

    if args.early_stop:
        save_path = ".\\"
        early_stopping = EarlyStopping(save_path)

    h = []
    for e in range(args.epochs):
        # forward
        model.train()
        h, beta_drug, beta_pro = model(train_g_pos_list, features)

        pos_score_0 = pred(train_pos_g_sum, h, e0)
        neg_score_0 = pred(train_neg_g_sum, h, e0)

        loss = compute_loss_xx(pos_score_0, neg_score_0)
        train_auc = compute_roc_auc_xx(pos_score_0, neg_score_0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        val_loss, val_auc = evalute_xx(model, h, val_g_pos_list, val_pos_g_sum, val_neg_g_sum, e_list)
        if args.early_stop:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break

        print('In epoch {:05d} | loss: {} | train_auc: {} | val_loss: {} | val_auc: {}'.format(e, loss, train_auc,
                                                                                                     val_loss, val_auc))
    if args.early_stop:
        model.load_state_dict(torch.load('best_network.pth'))
    with torch.no_grad():
        pos_score = pred(test_pos_g_sum, h, e0)
        neg_score = pred(test_neg_g_sum, h, e0)

        roc_auc_score = compute_roc_auc_xx(pos_score, neg_score)
        aupr = compute_aupr(pos_score, neg_score)
        acc, f1 = compute_acc_f1(pos_score, neg_score)

        print('AUC', roc_auc_score)
        print('AUPR', aupr)
        print('Acc', acc)
        print('F1', f1)

if __name__ == '__main__':
    # ----------- Parameter settings ------------------------ #
    parser = argparse.ArgumentParser(description='Train for link prediction')
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=30,
                        help="indicates whether to use early stop or not")

    args = parser.parse_args()
    print(args)
    main(args)
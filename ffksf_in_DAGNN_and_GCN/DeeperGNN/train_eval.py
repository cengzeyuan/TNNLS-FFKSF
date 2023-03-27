from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import *
import networkx as nx
from tqdm import tqdm

#multi-metrics
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

seed = 20
torch.manual_seed(seed)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def ffkgcn_preprocess_adj(adj, k, gamma):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1), dtype=np.float32)
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_add = adj
    k_adj = np.power(gamma, k) * sp.eye(adj.shape[0]) + np.power(gamma, k-1) * adj
    for i in range(k-1):
        adj_add = sp.csr_matrix(adj_add).dot(d_mat_inv_sqrt).dot(adj)
        k_adj += np.power(gamma, k-2-i)*adj_add
    adj_normalized = normalize_adj(k_adj)

    return adj_normalized.A

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, lcc_mask):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data

def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        k, gamma, device, permute_masks=None,  logger=None, lcc=False, save_path=None):

    val_losses, accs, accuracys, pres, recalls, f1s, durations = [], [], [], [], [], [], []
    lcc_mask = None
    if lcc:  # select largest connected component
        print("dataset:", dataset)
        data_ori = dataset[0]
        print("data_ori:", data_ori)
        data_nx = to_networkx(data_ori)
        print("data_nx:", data_nx)
        data_nx = data_nx.to_undirected()
        print("data_nx.to_undirected:", data_nx)
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)


  
    data = dataset[0]
    pbar = tqdm(range(runs), unit='run')
    #renormalization
    num_nodes = int(data.edge_index.max()) + 1
    adj = np.zeros([num_nodes, num_nodes])
    for item in data.edge_index.t().tolist():
        a = item[0]
        b = item[1]
        if a == b:
            continue
        else:
            adj[a, b] += 1
    re_adj = ffkgcn_preprocess_adj(adj, k, gamma)
    edg = list()
    weight = list()
    for i in range(re_adj.shape[0]):
        for j in range(re_adj.shape[1]):
            if re_adj[i, j] > 0:
                edg.append((i, j))
                weight.append(re_adj[i, j])
    edge_index = torch.LongTensor(edg).t()
    norm = torch.FloatTensor(weight)
    data.edge_index = edge_index
    data.edge_attr = norm

    for _ in pbar:
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, lcc_mask)
        data = data.to(device)

        model.to(device)
        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        #multi
        test_accuracy = 0
        test_pre = 0
        test_recall = 0
        test_f1 = 0

        val_loss_history = []

        for epoch in range(1, epochs + 1):
            out = train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
                #multi
                test_accuracy = eval_info['test_accuracy']
                test_pre = eval_info['test_pre']
                test_recall = eval_info['test_recall']
                test_f1 = eval_info['test_f1']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        accuracys.append(test_accuracy)
        pres.append(test_pre)
        recalls.append(test_recall)
        f1s.append(test_f1)

        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    accuracy, pre, recall, f1 = tensor(accuracys), tensor(pres), tensor(recalls), tensor(f1s)
    print('order: {:.3f},gamma: {:.3f},Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(k,
                 gamma,
                 loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    # multi
    print( 'Test Accuracy: {:.3f} ± {:.3f},Test pre: {:.3f} ± {:.3f},Test recall: {:.3f} ± {:.3f},Test f1: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(accuracy.mean().item(),
                 accuracy.std().item(),
                 pre.mean().item(),
                 pre.std().item(),
                 recall.mean().item(),
                 recall.std().item(),
                 f1.mean().item(),
                 f1.std().item(),
                 duration.mean().item()))

def train(model, optimizer, data, norm = None):
    model.train()
    optimizer.zero_grad()
    out = model(data, norm)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()



def evaluate(model, data, norm = None):
    model.eval()

    with torch.no_grad():
        logits = model(data, norm)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
        #mutil
        outs['{}_accuracy'.format(key)] = accuracy_score(data.y[mask].cpu(), pred.cpu())
        outs['{}_pre'.format(key)] = precision_score(data.y[mask].cpu(), pred.cpu(), average='macro')
        outs['{}_recall'.format(key)] = recall_score(data.y[mask].cpu(), pred.cpu(), average='macro')
        outs['{}_f1'.format(key)] = f1_score(data.y[mask].cpu(), pred.cpu(), average='macro')

    return outs

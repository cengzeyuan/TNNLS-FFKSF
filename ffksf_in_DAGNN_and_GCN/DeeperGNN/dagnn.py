import argparse
import torch
import os
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from train_eval import *
from datasets import *

import warnings


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--gamma', type=float, default=2)
parser.add_argument('--cuda', type=int, default=3)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)

    def forward(self, x, edge_index, norm, edge_weight=None):

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()

class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = Prop(dataset.num_classes, args.K)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data, norm):
        x, edge_index, norm = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, norm)
        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes, add_self_loops=False, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data, norm):
        x, edge_index, norm = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight=norm).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=norm)
        return F.log_softmax(x, dim=1)


class PropI(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(PropI, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, norm, edge_weight=None):
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)

class GCNI(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNI, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = PropI(dataset.num_classes, args.K)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, norm = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin1(x).relu()
        # x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, norm)
        # x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class GCN_one_layer(torch.nn.Module):
    def __init__(self, dataset):
        super(GCN_one_layer, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, dataset.num_classes, add_self_loops=False, normalize=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, data, norm):
        x, edge_index, norm = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight=norm)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:' + str(args.cuda) if (torch.cuda.is_available() and int(args.cuda) >= 0) else 'cpu')

warnings.filterwarnings("ignore", category=UserWarning)
print(args)
if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    print("dataset:", dataset)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset[0])
    run(dataset, DAGNN(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.k, args.gamma, permute_masks, lcc=False)
elif args.dataset == "cs" or args.dataset == "physics":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    run(dataset, GCN(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.k, args.gamma, device, permute_masks, lcc=False)






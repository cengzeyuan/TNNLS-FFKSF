from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from model import *
import uuid
#multi-metrics
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=3, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--gamma', type=float, default=2, help='gamma.')
parser.add_argument('--K', type=int, default=3, help='approximation order.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--runs', type=int, default=100, help='repeated.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data, args.K, args.gamma)
cudaid = "cuda:" + str(args.dev) if torch.cuda.is_available() else 'cpu'
print(cudaid)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

model = GCNII(nfeat=features.shape[1],
              nlayers=args.layer,
              nhidden=args.hidden,
              nclass=int(labels.max()) + 1,
              dropout=args.dropout,
              lamda=args.lamda,
              alpha=args.alpha,
              variant=args.variant).to(device)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(), acc_val.item()


def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #multi-metrics
        y_true = labels[idx_test].cpu()
        y_preds = output[idx_test].max(1)[1].type_as(y_true).cpu()
        accuracy_test = accuracy_score(y_true, y_preds)
        precision_test = precision_score(y_true, y_preds, average='macro')
        recall_test = recall_score(y_true, y_preds, average='macro')
        f1_test = f1_score(y_true, y_preds, average='macro')
        return loss_test.item(), acc_test.item(), accuracy_test.item(), precision_test.item(), recall_test.item(), f1_test.item()

run = args.runs
test_acc = np.zeros(run)
test_accuracy = np.zeros(run)
test_predictions = np.zeros(run)
test_recalls = np.zeros(run)
test_f1s = np.zeros(run)
for i in range(run):
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc, accuracys, prediction, recall, f1 = 0, 0, 0, 0, 0
    model.reset_parameters()
    optimizer = optim.Adam([
        {'params': model.params1, 'weight_decay': args.wd1},
        {'params': model.params2, 'weight_decay': args.wd2},
    ], lr=args.lr)
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train()
        loss_val, acc_val = validate()
        # if (epoch + 1) % 1 == 0:
            # print('Epoch:{:04d}'.format(epoch + 1),
            #       'train',
            #       'loss:{:.3f}'.format(loss_tra),
            #       'acc:{:.2f}'.format(acc_tra * 100),
            #       '| val',
            #       'loss:{:.3f}'.format(loss_val),
            #       'acc:{:.2f}'.format(acc_val * 100))
        if loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        _, acc, accuracys, prediction, recall, f1 = test()

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100))
    if args.test:
        test_acc[i] += acc
        test_accuracy[i] += accuracys
        test_predictions[i] += prediction
        test_recalls[i] += recall
        test_f1s[i] += f1
print(np.mean(test_acc))
print(np.std(test_acc))
print('Test Accuracy: {:.3f} ± {:.3f},Test pre: {:.3f} ± {:.3f},Test recall: {:.3f} ± {:.3f},Test f1: {:.3f} ± {:.3f}'.
          format(np.mean(test_accuracy),
                 np.std(test_accuracy),
                 np.mean(test_predictions),
                 np.std(test_predictions),
                 np.mean(test_recalls),
                 np.std(test_recalls),
                 np.mean(test_f1s),
                 np.std(test_f1s)
                 ))
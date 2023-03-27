import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy, all_metrics
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented
#args.weight_decay = 0.0000235455872331823
print(args)
# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.K, args.gamma, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC": features, precompute_time = sgc_precompute(features, adj, args.degree)
print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter()-t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time

# def test_regression(model, test_features, test_labels):
#     model.eval()
#     return accuracy(model(test_features), test_labels)

def test_regression(model, test_features, test_labels):
    model.eval()
    return all_metrics(model(test_features), test_labels)

# accura = np.zeros(100)
run = 10
test_accuracy = np.zeros(run)
test_predictions = np.zeros(run)
test_recalls = np.zeros(run)
test_f1s = np.zeros(run)

for i in range(run):

    if args.model == "SGC":
        model.reset_parameters
        model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],
                         args.epochs, args.weight_decay, args.lr, args.dropout)
        acc_test, prediction_test, recall_test, f1_test = test_regression(model, features[idx_test], labels[idx_test])

    print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
    print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))
    test_accuracy[i] += acc_test
    test_predictions[i] += prediction_test
    test_recalls[i] += recall_test
    test_f1s[i] += f1_test
# print(np.mean(accura),np.std(accura))
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
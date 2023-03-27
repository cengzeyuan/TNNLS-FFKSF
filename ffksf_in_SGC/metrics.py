from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    macro = f1_score(labels, preds, average='macro')
    return micro, macro

def all_metrics(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    # micro = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    prediction = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    macro_f1 = f1_score(labels, preds, average='macro')
    return acc, prediction, recall, macro_f1
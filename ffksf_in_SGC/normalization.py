import numpy as np
import scipy.sparse as sp
import torch

def aug_normalized_adjacency(adj, K, gamma):

    # adj = sp.coo_matrix(adj)
    # row_sum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # adj = 1*adj + 1*sp.eye(adj.shape[0]) + 1*d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    gamma = gamma
    k = K
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    w = np.ones(k + 1)
    print(w)
    for i in range(k + 1):
        w[i] *= np.power(gamma, k - i)
        # w[i] *= math.exp(k-i)
    print(w)
    adj_add = adj
    k_adj = w[0] * sp.eye(adj.shape[0]) + w[1] * adj
    for i in range(k - 1):
        adj_add = adj_add.dot(d_mat_inv_sqrt).dot(d_mat_inv_sqrt).dot(adj)
        k_adj += w[i + 2] * adj_add
        print(i)
    k_adj = sp.coo_matrix(k_adj)
    row_sum = np.array(k_adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(k_adj).dot(d_mat_inv_sqrt).tocoo()

    # adj = adj + sp.eye(adj.shape[0])
    # adj = sp.coo_matrix(adj)
    # row_sum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def fetch_normalization(type):
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
   }
   func = switcher.get(type, lambda: "Invalid normalization technique.")
   return func

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# TNNLS-FFKGCN
==================================================

## 1. Overview

This is an implementation of "Graph Neural Networks with High-Order Polynomial Spectral Filters" [(link)](https://ieeexplore.ieee.org/document/10098821), a superior substitution of the popular 1st-order Chebyshev filter.

It’s commonly believed that deep graph neural networks will be over-smoothing resulting from the Laplacian smoothing. However, we demonstrate that graph neural networks with residual connections or with high-order polynomial spectral filters could be kept from performance degradation with the extension of the receptive fields.

Furthermore, we demonstrate the generalized renormalization trick is scale-invariant and propose a reversible transformation for the domain of polynomial coefficients. With a forgetting factor, the optimization of coefficients is transformed to the tuning of a hyperparameter and a novel spectral filter is designed, working as a superior substitution of the popular 1st-order Chebyshev filter.

We further complete this filter as a specific generalized renormalization trick and apply it to five models, namely GCN, SGC, GCNII, DAGNN and HGCN. These models are implemented according to the codes publicly relseased, except for the following minor changes in DAGNN and GCNII: for DAGNN, we set \gamma = 1 and K = 1 in the generalized renormalization trick; For GCNII, we further complete the missing step of resetting parameters.

One aspect still needs improving in the implementation based on PyTorch Geometric. Due to the unique data form of adjacency matrices in PyTorch Geometric, it takes a lot of time to do the data transformation as in FFK-DAGNN. We would really appreciate any method proposed to deal with this inefficiency. However, if the dataset is loaded manually, for instance, GCNII, it is advisable to directly apply this renormalization trick.

For more insights, (empirical and theoretical) analysis, and discussions, please refer to our paper.

Note that the implementation of the renormalization trick is not in the model.py file.

Thank you for reading this far.

## 2. Requirement

Models built based on different libraries may have different performances. Therefore, we use the models the authors release. We retain the requirement part in the original README.md file of each project to help the readers get the corresponding requirements more conveniently.

## 3. Concluded as a trick

We have distilled the algorithm into a simple and practical trick as follows.

If the dataset is loaded manually, for instance, GCNII, it is advisable to directly apply this renormalization trick.
```
def ffk_normalized_adjacency(adj, k, gamma):
   # adj = sys_normalized_adjacency(adj)
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   row_sum = (row_sum == 0) * 1 + row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   # k = 3
   # gamma = 2
   w = np.ones(k + 1)
   print(w)
   for i in range(k + 1):
       w[i] *= np.power(gamma, k - i)
   print(w)
   adj_add = adj
   k_adj = w[0] * sp.eye(adj.shape[0]) + w[1] * adj
   for i in range(k - 1):
       adj_add = adj_add.dot(d_mat_inv_sqrt).dot(d_mat_inv_sqrt).dot(adj)
       k_adj += w[i + 2] * adj_add
       print(i)
   row_sum = np.array(k_adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(k_adj).dot(d_mat_inv_sqrt).tocoo()
```

Due to the unique data form of adjacency matrices in PyTorch Geometric, it should takes a lot of time to do the data transformation. Here is an example.
```
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
```

## 4. Examples

Please enter the directory based on the model to your interest and follow the instructions below:

FFK-DAGNN or FFK-GCN:   (directory: ffksf_in_DAGNN_and_GCN)

```python dagnn.py --dataset=physics --weight_decay=0 --K=5 --dropout=0.8  --gamma 5 --k 3 --cuda 2 --runs 100```

FFK-HGCN or FFK-GCN:    (directory: ffksf_in_HGCN_and_GCN)

```python train.py --task nc --dataset airport --model FFKGCN --lr 0.01 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 2 --dim 256 --num-layers 4```

FFK-GCNII:              (directory: ffksf_in_GCNII)

```python train.py --data cora --layer 128 --gamma 3 --K 6 --dev 2```

FFK-SGC:                (directory: ffksf_in_SGC)

```python citation.py --dataset cora --tuned --gamma 2 --K 7```

If you are interested in the detailed settings of hyperparameters, please refer to our paper. For FFK-GCNII and FFK-DAGNN, we strictly follow the settings of hyperparameters of GCNII and DAGNN respectively. Thus, we retain the run.sh file and the semi.sh file in the original implementation of GCNII and DAGNN.

## 5. The Codes of Original Models

Thanks very much for the following implementations:

[1] HGCN: https://github.com/HazyResearch/hgcn

[2] SGC: https://github.com/Tiiiger/SGC

[3] DAGNN: https://github.com/divelab/DeeperGNN

[4] GCNII: https://github.com/chennnM/GCNII

## 6. Citation

If you find this code useful or this article useful, please cite the following paper:
```
@ARTICLE{10098821,
  author={Zeng, Zeyuan and Peng, Qinke and Mou, Xu and Wang, Ying and Li, Ruimeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Graph Neural Networks With High-Order Polynomial Spectral Filters}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2023.3263676}}
```
## 7. Something Missing

Limited by the size of files in GitHub, we did not upload all the pretrained files of FFK-GCNII. If you are interested in it, please do not hesitate to contact us.

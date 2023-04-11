# TNNLS-FFKGCN
==================================================

## 1. Overview

This is an implementation of "Graph Neural Networks with High-Order Polynomial Spectral Filters", a superior substitution of the popular 1st-order Chebyshev filter.

It’s commonly believed that deep graph neural networks will be over-smoothing resulting from the Laplacian smoothing. However, we demonstrate that graph neural networks with residual connections or with high-order polynomial spectral filters could be kept from performance degradation with the extension of the receptive fields.

Furthermore, we demonstrate the generalized renormalization trick is scale-invariant and propose a reversible transformation for the domain of polynomial coefficients. With a forgetting factor, the optimization of coefficients is transformed to the tuning of a hyperparameter and a novel spectral filter is designed, working as a superior substitution of the popular 1st-order Chebyshev filter.

We further complete this filter as a specific generalized renormalization trick and apply it to five models, namely GCN, SGC, GCNII, DAGNN and HGCN. These models are implemented according to the codes publicly relseased, except for the following minor changes in DAGNN and GCNII: for DAGNN, we set \gamma = 1 and K = 1 in the generalized renormalization trick; For GCNII, we further complete the missing step of resetting parameters.

One aspect still needs improving in the implementation based on PyTorch Geometric. Due to the unique data form of adjacency matrices in PyTorch Geometric, it takes a lot of time to do the data transformation as in FFK-DAGNN. We would really appreciate any method proposed to deal with this inefficiency. However, if the dataset is loaded manually, for instance, GCNII, it is advisable to directly apply this renormalization trick.

For more insights, (empirical and theoretical) analysis, and discussions, please refer to our paper.

Note that the implementation of the renormalization trick is not in the model.py file.

Thank you for reading this far.

## 2. Requirement

Models built based on different libraries may have different performances. Therefore, we use the models the authors release. We retain the requirement part in the original README.md file of each project to help the readers get the corresponding requirements more conveniently.

## 3. Examples

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

## 4. The Codes of Original Models

Thanks very much for the following implementations:

[1] HGCN: https://github.com/HazyResearch/hgcn

[2] SGC: https://github.com/Tiiiger/SGC

[3] DAGNN: https://github.com/divelab/DeeperGNN

[4] GCNII: https://github.com/chennnM/GCNII

## 5. Citation

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
## 6. Something Missing

Limited by the size of files in GitHub, we did not upload all the pretrained files of FFK-GCNII. If you are interested in it, please do not hesitate to contact us.

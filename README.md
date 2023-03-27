# TNNLS-FFKGCN
==================================================

## 1. Overview

This is an implementation of "Graph Neural Networks with High-Order Polynomial Spectral Filters".

There is a mainstream view that deep graph neural networks tend to be smoothing resulting from the Laplacian smoothing. However, in this article, we demonstrate that graph neural networks with residual connections or with high-order polynomial spectral filters could be kept from performance degradation with the extension of the receptive fields.

Further, we demonstrate the generalized renormalization trick is scale-invariant and propose a reversible transformation for the domain of polynomial coefficients. Afterward, the optimization of coefficients is transformed to the tuning of a hyperparameter and a novel spectral filter is designed, working as a superior substitution of the popular 1st-order Chebyshev filter.

Herein, we further complete this filter as a generalized renormalization trick and apply it to five models, namely GCN, SGC, GCNII, DAGNN and HGCN. These models are implemented according to the ones the authors release, except for the following minor changes in DAGNN and GCNII. For DAGNN, we set \gamma = 1 and K = 1 in the generalized renormalization trick. For GCNII, we further complete the missing step of resetting parameters.

There is still a little trouble in the implementation based on PyTorch Geometric. Due to the unique data form in PyTorch Geometric, we have to do a transformation such as the one in FFK-DAGNN, which takes a lot of time. If there is any method that can deal with it, please do not hesitate to contact us. However, if the implementation is based on DGL or TensorFlow, it is OK to directly apply this renormalization trick.

For more insights, (empirical and theoretical) analysis, and discussions, please refer to our paper.

Thank you for reading this far.

## 2. Requirement

Models built based on different libraries may have different performance. Therefore, we use the models the authors release. We retain the requirement part in the original README.md file of each project to help the readers get the corresponding requirements more convenient.

## 3. Examples

Please enter the file where the model you prefer and follow the instructions below:

FFK-DAGNN or FFK-GCN:   (file: ffksf_in_DAGNN_and_GCN)

```python dagnn.py --dataset=physics --weight_decay=5e-4 --epochs 200 --early_stopping 100 --hidden 64 --gamma 5 --k 3 --cuda 2 --dropout 0.5 --runs 100```

FFK-HGCN or FFK-GCN:    (file: ffksf_in_HGCN_and_GCN)

```python -u train.py --task nc --dataset airport --model FFKGCN --lr 0.01 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold Euclidean --log-freq 5 --cuda 2 --dim 256 --num-layers 4```

FFK-GCNII:              (file: ffksf_in_GCNII)

```python train.py --data cora --layer 128 --gamma 3 --K 6 --dev 2```

FFK-SGC:                (file: ffksf_in_SGC)

```python citation.py --dataset cora --tuned --gamma 2 --K 7```

## 4. The Codes of Original Models

Thanks very much for the following implementation:

[1] HGCN: https://github.com/HazyResearch/hgcn

[2] SGC: https://github.com/Tiiiger/SGC

[3] DAGNN: https://github.com/divelab/DeeperGNN

[4] GCNII: https://github.com/chennnM/GCNII

## 5. Citation

If you find this code useful or this article useful, please cite the following paper:

## 6. Lack

Limited by the size of file in github, we did not upload all the pretrained files of FFK-GCNII. If it is interesing for you, please do not hesitate to contact us.

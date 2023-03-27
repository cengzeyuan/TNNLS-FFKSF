## Requirements
* PyTorch
* PyTorch Geometric >= 1.3.1  
* NetworkX
* tqdm  


Note that the versions of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which need to be installed in advance. It would be easy by following the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).    

~~PyTorch Geometric 1.3.1 was used in this code. If you have a newer version installed already, you may encounter an error about "GCNConv.norm" when running this code. Refer to this [issue](https://github.com/mengliu1998/DeeperGNN/issues/2) for a possible solution.~~ (2020.8.12 update: This issue has been solved in the current code. Now, our code works for PyTorch Geometric >= 1.3.1.)

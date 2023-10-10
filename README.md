# MTGCL: Multi-Task Graph Contrastive Learning for Identifying Cancer Driver Genes from Multi-omics Data

MTGCL is a multi-task learning model framework that introduces the graph contrastive learning to design the auxiliary task and proposes a novel graph convolutional layer structure for the main task of GCN-based node classification.

## Requirements
The project is written in Python 3.7 and all experiments were conducted on the Ubuntu server with an Intel Xeon CPU (2.4GHz, 128G RAM) and an Nvidia RTX 3090 GPU (24G GPU RAM). For the faster training process, training on a GPU is necessary, but a standard computer without GPU also works (consuming much more training time). 
We recommend the hardware configuration as follows:

- CPU RAM >= 16G
- GPU RAM >= 8G

All implementations of MTGCL and the GNN-based baselines were based on PyTorch and PyTorch Geometric. MTGCL requires the following dependencies:

- python == 3.7.1
- numpy == 1.20.2
- pandas == 1.2.5
- scikit-learn == 0.24.2
- scipy == 1.7.0
- pytorch == 1.7.1
- torch-geometric == 1.7.2


## Reproducibility
The scripts of `main.py` can reproduce the comparison results, which can be done by the following commands, taking the CPDB network as an example, the hyperparameters corresponding to each method can be set in the utils script:

```
python main.py
```





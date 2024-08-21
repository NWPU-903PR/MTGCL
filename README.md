# MTGCL: Multi-Task Graph Contrastive Learning for Identifying Cancer Driver Genes from Multi-omics Data

MTGCL proposes a novel graph convolutional layer structure, effectively integrating graph structure topology information and node features information, and employs a semi-supervised graph contrastive learning task as a regularizer within a multi-task learning paradigm. 

## Requirements
The project is written in Python 3.7 and all experiments were conducted on the Ubuntu server with an Intel Xeon CPU (2.4GHz, 128G RAM) and an Nvidia RTX 3090 GPU (24G GPU RAM). For the faster training process, training on a GPU is necessary, but a standard computer without GPU also works (consuming much more training time). 
We recommend the hardware configuration as follows:

- CPU RAM >= 16G
- GPU RAM >= 12G

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
python main.py -cancer "CPDB" -model "MTGCL" -e 1900 -lr 0.001 -hd [300,100] -dropout 0.5 - dropout_edge 0.5 -pe1 0.1 -pe2 0.3 -pf1 0.1 -pf2 0.5 -t 0.5 -lamuta 0.7 -cv 5 -num 10
```
The parameter "-cancer" can take a range of values including pan-cancer "CPDB", "STRING", "PathNet", and specific-type cancers such as "BRCA", "LIHC", "COAD", "PRAD", "UCEC". The comparison methods and baseline methods can be changed by modifying the parameter "-model", and adjusting the corresponding hyperparameters according to  Supplementary A5. The datasets  are also provided in the "./data/data_paper" folder.





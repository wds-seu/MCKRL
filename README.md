# MCKRL: A Multi-Graph Knowledge Representation Learning Model

## Requirements
conda env: ./conda_environment.yaml

pip package: ./requirements.txt

## Datasets
Due to the large size of the dataset, only a list of links is provided for download. The required links are as follows:

DrugBank: https://go.drugbank.com/releases/5-1-7\#full

UniPortKB: https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_01/

BioSE: http://snap.stanford.edu/biodata/

## Main File
./dataset: contain processed datasets

./attention.py: implementation of attention mechanism
./early_stopping.py: early stopping to prevent overfitting

./main.py: the main execution function, including the training process

./makeDataset.py: convert raw files to .dgl dataset

./model.py: the main execution function, including the training process

./predictor.py: dot product operation

./processDrugBank: process the .xml of DrugBank

./utils.py: contain some utility functions

./vocab.py: convert a list of nodes ids to vocab ids

## Run
`python main.py`

## Related Work
GCN: Semi-aupervised Classification with Graph Convolutional Networks [paper](https://arxiv.org/abs/1609.02907) [code](https://github.com/tkipf/pygcn)

GAT: Graph Attention Networks [paper](https://arxiv.org/abs/1710.10903) [code](https://github.com/Diego999/pyGAT/)

Decagon: Modeling polypharmacy side effects with graph convolutional networks [paper](https://academic.oup.com/bioinformatics/article-abstract/34/13/i457/5045770) [code](https://github.com/marinkaz/decagon)

R-GCN: Modeling Relational Data with Graph Convolutional Networks [paper](https://link.springer.com/chapter/10.1007/978-3-319-93417-4_38) [code](https://github.com/tkipf/relational-gcn)

RSN: Learning to exploit long-term relational dependencies in knowledge
graphs [paper](http://proceedings.mlr.press/v97/guo19c.html?ref=https://githubhelp.com) [code](https://github.com/nju-websoft/RSN)

AM-GCN: Adaptive Multi-channel Graph Convolutional Networks [paper](https://dl.acm.org/doi/abs/10.1145/3394486.3403177) [code](https://github.com/zhumeiqiBUPT/AM-GCN)

HetGNN: Heterogeneous Graph Neural Network [paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330961) [code](https://github.com/Jhy1993/HAN)




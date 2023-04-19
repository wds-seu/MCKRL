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
GCN: paper code{https://github.com/tkipf/pygcn}
GAT: code{https://github.com/Diego999/pyGAT/}
Decagon: code{https://github.com/marinkaz/decagon}


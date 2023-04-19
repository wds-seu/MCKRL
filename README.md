# MCKRL: A Multi-Graph Knowledge Representation Learning Model

## Requirements
conda env: ./conda_environment.yaml

pip package: ./pip_packages.txt

## Datasets
Due to the large size of the dataset, only a list of links is provided for download. The required links are as follows:
DrugBank: https://go.drugbank.com/releases/5-1-7\#full
UniPortKB: https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2021_01/
BioSE: http://snap.stanford.edu/biodata/

## Main File
./early_stopping.py: early stopping to prevent overfitting
./main.py: the main execution function, including the training process
./makeDataset.py: convert raw files to .dgl dataset
./model.py: the main execution function, including the training process
./predictor.py: dot product operation
./utils.py: contain some utility functions
./vocab.py: convert a list of nodes ids to vocab ids

## Run
`python main.py`

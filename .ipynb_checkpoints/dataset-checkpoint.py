import torch
from torch.utils.data import Dataset

import os
import pandas as pd

from utils import fetch_params, to_hot

def make_data(DATADIR, n=None):
    
    dataspec = fetch_params(os.path.join(DATADIR, 'spec.json'))
    
    train_file = os.path.join(DATADIR, 'train.smi')
    test_file = os.path.join(DATADIR, 'test.smi')
    
    traindf = pd.read_csv(train_file)
    testdf = pd.read_csv(test_file)
    
    if n:
        traindf = traindf.iloc[:n]
        testdf = testdf.iloc[:n]

    # Make Train Tensors
    
    train_hots = to_hot(
        traindf['smiles'], 
        stoi=dataspec.stoi, 
        smile_len=dataspec.smile_len
    )
    
    train_props = torch.tensor(traindf.iloc[:, 1:].values, dtype=torch.float32)
    train_props = torch.tensor_split(train_props, train_props.size(1), dim=1)
    
    # Make Test Tensors
    
    test_hots = to_hot(
        testdf['smiles'], 
        stoi=dataspec.stoi, 
        smile_len=dataspec.smile_len
    )
    
    test_props = torch.tensor(testdf.iloc[:, 1:].values, dtype=torch.float32)
    test_props = torch.tensor_split(test_props, test_props.size(1), dim=1)
    
    # Make Datasets
    
    trainset = SmileDataset(train_hots, *train_props)
    testset = SmileDataset(test_hots, *test_props)
    
    return trainset, testset

class SmileDataset(Dataset):
    def __init__(self, hots, logp, qed, sas):
        self.hots = hots
        self.logp = logp
        self.qed = qed
        self.sas = sas

    def __len__(self):
        return self.hots.size(0)
    
    def __getitem__(self, idx):
        return self.hots[idx], self.logp[idx], self.qed[idx], self.sas[idx]
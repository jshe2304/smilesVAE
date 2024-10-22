import torch
from torch.utils.data import Dataset

import random
import os
import pandas as pd

def make_data(DATADIR, to_tensor, device='cpu', n=None):

    train_file = os.path.join(DATADIR, 'train.smi')
    test_file = os.path.join(DATADIR, 'test.smi')
    
    traindf = pd.read_csv(train_file)
    testdf = pd.read_csv(test_file)
    
    if n is not None:
        traindf = traindf.sample(n)
        testdf = testdf.sample(n)

    # Make Train Tensors
    
    train_hots = to_tensor(traindf['smiles'])
    
    train_props = torch.tensor(traindf.iloc[:, 1:].values, dtype=torch.float32)
    train_props = torch.tensor_split(train_props, train_props.size(1), dim=1)
    
    # Make Test Tensors
    
    test_hots = to_tensor(testdf['smiles'])
    
    test_props = torch.tensor(testdf.iloc[:, 1:].values, dtype=torch.float32)
    test_props = torch.tensor_split(test_props, test_props.size(1), dim=1)
    
    # Make Datasets
    
    trainset = SmileDataset(train_hots, *train_props, device)
    testset = SmileDataset(test_hots, *test_props, device)
    
    return trainset, testset

class SmileDataset(Dataset):
    def __init__(self, hots, logp, qed, sas, device):
        self.hots = hots
        self.logp = logp
        self.qed = qed
        self.sas = sas
        self.device = device
        self.n = self.hots.size(0)

    def __len__(self):
        return self.n
    
    def sample(self, n):
        indices = random.sample(range(self.n), n)
        
        return (self.hots[indices].to(self.device), 
                self.logp[indices].to(self.device), 
                self.qed[indices].to(self.device), 
                self.sas[indices].to(self.device))
    
    def __getitem__(self, idx):
        device = self.device
        return self.hots[idx].to(device), self.logp[idx].to(device), self.qed[idx].to(device), self.sas[idx].to(device)
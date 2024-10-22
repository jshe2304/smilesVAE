import torch
import torch.nn as nn


class Predictor(nn.Module):
    '''
    Dense property predictor with dropout. 
    '''
    def __init__(self, L, hidden_size=48, *args, **kwargs):
        super().__init__()
        
        # Dense Network with Dropout
        self.dense = nn.Sequential(
            nn.Linear(L, L), 
            nn.ReLU(), 
            nn.Dropout(0.15), 
            
            nn.Linear(L, hidden_size), 
            nn.ReLU(), 
            nn.Dropout(0.15)
        )
        
        # LogP Perceptron
        self.logp = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
        # QED Perceptron
        self.qed = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        
        # SAS Perceptron
        self.sas = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, latent):
        latent = self.dense(latent)
        
        return self.logp(latent), self.qed(latent), self.sas(latent)
        
        
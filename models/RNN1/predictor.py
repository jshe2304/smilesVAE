import torch
import torch.nn as nn

H = 64

class Predictor(nn.Module):
    '''
    Dense property predictor with dropout. 
                  
                 --> logp Linear
    Dense Layers --> qed Linear
                 --> sas Linear
    '''
    def __init__(self, L):
        super().__init__()
        
        # Dense Dropout Network
        
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=L, 
                out_features=L
            ), 
            nn.ReLU(), 
            
            nn.Dropout(0.15), 
            
            nn.Linear(
                in_features=L, 
                out_features=H
            ), 
            nn.ReLU(), 
            
            nn.Dropout(0.15), 
            nn.BatchNorm1d(H)
        )
        
        # LogP Perceptron
        
        self.logp = nn.Sequential(
            nn.Linear(
                in_features=H, 
                out_features=1
            )
        )
        
        # QED Perceptron
        
        self.qed = nn.Sequential(
            nn.Linear(
                in_features=H, 
                out_features=1
            )
        )
        
        # SAS Perceptron
        
        self.sas = nn.Sequential(
            nn.Linear(
                in_features=H, 
                out_features=1
            )
        )

    def forward(self, latent):
        latent = self.dense(latent)
        
        return self.logp(latent), self.qed(latent), self.sas(latent)
        
        
import torch
import torch.nn as nn

class Predictor(nn.Module):
    '''
    Dense property predictor with dropout. 
                  
                 --> logp Linear
    Dense Layers --> qed Linear
                 --> sas Linear
    '''
    def __init__(self, modelspec, *args):
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=modelspec.latent_dim, 
                out_features=modelspec.latent_dim
            ), 
            nn.ReLU(), 
            
            nn.Dropout(modelspec.dropout), 
            
            nn.Linear(
                in_features=modelspec.latent_dim, 
                out_features=modelspec.pred_dim
            ), 
            nn.ReLU(), 
            
            nn.Dropout(modelspec.dropout), 
            nn.BatchNorm1d(modelspec.pred_dim)
        )
        
        self.logp = nn.Sequential(
            nn.Linear(
                in_features=modelspec.pred_dim, 
                out_features=1
            )
        )
        
        self.qed = nn.Sequential(
            nn.Linear(
                in_features=modelspec.pred_dim, 
                out_features=1
            )
        )
        
        self.sas = nn.Sequential(
            nn.Linear(
                in_features=modelspec.pred_dim, 
                out_features=1
            )
        )

    def forward(self, latent):
        latent = self.dense(latent)
        
        return self.logp(latent), self.qed(latent), self.sas(latent)
        
        
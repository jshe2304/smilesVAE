import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params

        self.gru = nn.GRU(
            input_size=self.params.ALPHABET_LEN, 
            hidden_size=self.params.GRU_HIDDEN_DIM, 
            batch_first=True
        )
        
        self.dense_encoder = nn.Sequential(
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ), 
            nn.Tanh(), 
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.LATENT_DIM
            ), 
            nn.Tanh()
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        
        hidden = self.dense_encoder(hidden)
        
        return hidden
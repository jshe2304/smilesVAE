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
            
            nn.Dropout(p=0.1), 
            nn.BatchNorm1d(num_features=self.params.GRU_HIDDEN_DIM), 
            
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.LATENT_DIM
            ), 
            nn.Tanh(), 
            
            nn.Dropout(p=0.1), 
            nn.BatchNorm1d(num_features=self.params.LATENT_DIM), 
            
            nn.Linear(
                in_features=self.params.LATENT_DIM, 
                out_features=self.params.LATENT_DIM, 
            )
        )
        
        self.z_mean = nn.Linear(
            in_features=self.params.LATENT_DIM, 
            out_features=self.params.LATENT_DIM
        )
        
        self.z_logvar = nn.Linear(
            in_features=self.params.LATENT_DIM, 
            out_features=self.params.LATENT_DIM
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        
        hidden = self.dense_encoder(hidden.squeeze(0))
        
        # Latent Distribution
        
        z_mean = self.z_mean(hidden)
        
        z_logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=z_mean, device=z_mean.device)
        
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        
        return z_mean, z_logvar, z
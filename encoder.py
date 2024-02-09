import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    Compress token sequence into vector. Choice of CNN or RNN. 
    
    Token Sequence  -->  GRU/CNN  --> Dense  --> Variational Sampler  -->  Latent Vector
    '''
    
    def __init__(self, params, data, *args):
        super().__init__()
        
        # Workhorse RNN/CNN

        self.rnn = nn.GRU(
            input_size=len(data.alphabet), 
            hidden_size=params.hidden_dim, 
            batch_first=True, 
        )
            
        # Dense Layers
        
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=params.hidden_dim, 
                out_features=params.latent_dim, 
            ), 
            nn.ReLU(), 
            nn.Linear(
                in_features=params.latent_dim, 
                out_features=params.latent_dim
            )
        )
        
        # Distribution-Defining Dense Layers
        
        self.z_mean = nn.Linear(
            in_features=params.latent_dim, 
            out_features=params.latent_dim
        )
        
        self.z_logvar = nn.Linear(
            in_features=params.latent_dim, 
            out_features=params.latent_dim
        )

    def forward(self, x):
        
        # RNN and Linear Layers
        
        _, hidden = self.rnn(x)
        hidden = self.dense(hidden.squeeze(0))
        
        # Latent Distribution
        
        mean = self.z_mean(hidden)
        logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=logvar, device=logvar.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        
        return mean, logvar, z
import torch
import torch.nn as nn

H = 256 # hidden dimension
N = 2 # number of layers

class Encoder(nn.Module):
    '''
    Compress token sequence into vector. Choice of CNN or RNN. 
    
    Token Sequence  -->  GRU/CNN  --> Dense  --> Variational Sampler  -->  Latent Vector
    '''
    
    def __init__(self, L, alphabet_len=21):
        super().__init__()
        
        # Workhorse RNN/CNN
        
        self.rnn = nn.GRU(
            input_size=alphabet_len, 
            hidden_size=H, 
            batch_first=True, 
            num_layers=N, 
            dropout=0.1
        )
        
        self.dense = nn.Sequential(

            nn.Linear(
                in_features = N * H, 
                out_features = N * H
            ), 
            nn.ReLU(), 
            nn.Linear(
                in_features = N * H, 
                out_features = L
            )
        )
        
        # Distribution-Defining Dense Layers
        
        self.z_mean = nn.Linear(
            in_features = L, 
            out_features = L
        )
        
        self.z_logvar = nn.Linear(
            in_features = L, 
            out_features = L
        )
        
    def forward(self, x):
        
        # RNN and Linear Layers
        
        _, hidden = self.rnn(x)
        hidden = hidden.transpose(0, 1).flatten(start_dim=1)
        
        hidden = self.dense(hidden)

        # Latent Distribution
        
        mean = self.z_mean(hidden)
        logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=logvar)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        
        return mean, logvar, z
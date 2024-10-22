import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    Compress token sequence into vector. Choice of CNN or RNN. 
    
    Token Sequence  -->  GRU  --> Dense  --> Variational Sampler  -->  Latent Vector
    '''
    
    def __init__(self, L, H=256, N=2, alphabet_len=21, smile_len=40, arch='rnn'):
        super().__init__()
        
        # Workhorse RNN/CNN
        
        self.rnn = nn.GRU(
            input_size=alphabet_len, 
            hidden_size=H, 
            batch_first=True, 
            num_layers=N, 
            dropout=0.2
        )
        
        H_flat = N * H
        
        self.dense = nn.Sequential(
            
            nn.Flatten(start_dim=1),
            nn.Linear(
                in_features = H_flat, 
                out_features = H_flat
            ), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(
                in_features = H_flat, 
                out_features = L
            ), 
            nn.Dropout(0.1)
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
        hidden = hidden.transpose(0, 1)
        
        hidden = self.dense(hidden)

        # Latent Distribution
        
        mean = self.z_mean(hidden)
        logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=logvar)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        
        return mean, logvar, z

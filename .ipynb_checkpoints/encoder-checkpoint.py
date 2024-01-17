import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params

        self.gru = nn.GRU(
            input_size=self.params.ALPHABET_LEN, 
            hidden_size=self.params.GRU_HIDDEN_DIM, 
            batch_first=True, 
        )
        
        self.fc_network = nn.Sequential(
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.LATENT_DIM, 
            ), 
            nn.ReLU(), 
            nn.Linear(
                in_features=self.params.LATENT_DIM, 
                out_features=self.params.LATENT_DIM
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
        '''
        (Batch Size, Sequence Length, Alphabet Length)
         | gru
        (1, Batch Size, Hidden Size)
         | squeeze
        (Batch Size, Hidden Size)
         | dense sequential
        (Batch Size, Latent Size)
         | z_mean                   | z_logvar
        (Batch Size, Latent Size)  (Batch Size, Latent Size)
        '''
        
        # GRU and Linear Dense Layers
        
        _, hidden = self.gru(x)
        
        hidden = self.fc_network(hidden.squeeze(0))
        
        # Latent Distribution
        
        z_mean = self.z_mean(hidden)
        
        z_logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=z_logvar, device=z_logvar.device)
        
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        
        return z_mean, z_logvar, z
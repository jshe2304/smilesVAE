import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    Compress token sequence into vector. Choice of CNN or RNN. 
    
    Token Sequence  -->  GRU  --> Dense  --> Variational Sampler  -->  Latent Vector
    '''
    
    def __init__(self, L, hidden_size=384, alphabet_len=32, embedding_dim=8, *args, **kwargs):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=alphabet_len, 
            embedding_dim=embedding_dim
        )

        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, 9, padding=4), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Conv1d(embedding_dim, embedding_dim, 7, padding=3), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Conv1d(embedding_dim, embedding_dim, 5, padding=2), 
        )
        
        self.rnn = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.2
        )
        
        flat_hidden_size = 2 * hidden_size
        
        self.dense = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_hidden_size, flat_hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.1), 

            nn.Linear(flat_hidden_size, L), 
            nn.ReLU(), 
            nn.Dropout(0.1)
        )
        
        self.z_mean = nn.Linear(L, L)
        self.z_logvar = nn.Linear(L, L)

    def forward(self, idx):

        x = self.embed(idx)
        
        # CNN and RNN

        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.rnn(x)

        # Linear Layers
        
        hidden = hidden.transpose(0, 1)
        hidden = self.dense(hidden)

        # Latent Distribution
        
        mean = self.z_mean(hidden)
        logvar = self.z_logvar(hidden)
        
        # Latent Sample
        
        epsilon = torch.randn_like(input=logvar)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        
        return mean, logvar, z

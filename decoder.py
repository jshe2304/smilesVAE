import torch
import torch.nn as nn

class DecodeNext(nn.Module):
    '''
    Predict a single token from hidden state and last predicted token. 
    
    Input, Hidden  -->  GRU  -->  Dense  --> Logits
    '''
    
    def __init__(self, params, data, *args):
        super().__init__()
        
        # Workhorse RNN
        
        self.rnn = nn.GRU(
            input_size=len(data.alphabet), 
            hidden_size=params.hidden_dim, 
            batch_first=True, 
        )
        
        # Dense Layers
        
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=params.hidden_dim, 
                out_features=params.hidden_dim, 
            ), 
            nn.ReLU(),
            nn.Linear(
                in_features=params.hidden_dim, 
                out_features=len(data.alphabet)
            )
        )

    def forward(self, inp, hidden):

        out, hidden = self.rnn(inp, hidden)

        prediction = self.dense(out.squeeze(1))

        return prediction, hidden

class Decoder(nn.Module):
    '''
    Iteratively predict token sequence from provided hidden state. 
    
    Hidden  -->  Dense  -->  GRU  -->  Logits Sequence
    '''
    
    def __init__(self, params, data):
        super().__init__()
        
        self.data = data
        
        # Dense Layers
        
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=params.latent_dim, 
                out_features=params.hidden_dim
            ), 
            nn.ReLU(), 
            nn.Linear(
                in_features=params.hidden_dim, 
                out_features=params.hidden_dim
            )
        )
        
        # GRU Unit
        
        self.decode_next = DecodeNext(params, data)
    
    def forward(self, latent):
        
        # Dense Layers

        hidden = self.dense(latent).unsqueeze(0)
        
        # Prepare Sequence Tensor

        *_, batch_size, _ = hidden.shape
        
        y = torch.zeros(
            size=(batch_size, self.data.smile_len, len(self.data.alphabet)), 
            dtype=torch.float32
        )

        inp = torch.full(
            fill_value=-32,
            size=(batch_size, 1, len(self.data.alphabet)), 
            dtype=torch.float32
        )
        inp[:, :, self.data.stoi['<BOS>']] = 32
        y[:, 0, :] = inp.squeeze()
        
        # Iterative Token Prediction

        for i in range(1, self.data.smile_len):
            
            prediction, hidden = self.decode_next(inp, hidden)

            y[:, i, :] = prediction

            inp = prediction.unsqueeze(1)

        return y

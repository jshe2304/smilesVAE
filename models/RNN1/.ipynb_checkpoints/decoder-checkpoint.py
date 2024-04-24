import torch
import torch.nn as nn

class DecodeNext(nn.Module):
    '''
    Predict a single token from hidden state and last predicted token. 
    
    Input, Hidden  -->  GRU  -->  Dense  --> Logits
    '''
    
    def __init__(self, H_i, H_o, *args):
        super().__init__()
        
        # Workhorse RNN
        
        self.rnn = nn.GRU(
            input_size=H_i, 
            hidden_size=H_o, 
            batch_first=True, 
        )
        
        self.dense = nn.Linear(
            in_features = H_o, 
            out_features = H_i
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
    
    def __init__(self, L, H=256, alphabet_len=21, smile_len=40):
        super().__init__()
        
        self.alphabet_len = alphabet_len
        self.smile_len = smile_len
        
        # Dense Layers
        self.dense = nn.Sequential(
            nn.Linear(
                in_features=L, 
                out_features=H
            ), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(
                in_features=H, 
                out_features=H
            )
        )
        
        # GRU Unit
        self.decode_next = DecodeNext(alphabet_len, H)
    
    def forward(self, latent, target=None):
        
        # Dense Layers
        hidden = self.dense(latent).unsqueeze(0)
        
        # Sequence Tensor
        _, batch_size, _ = hidden.shape
        y = torch.zeros(
            size=(batch_size, self.smile_len, self.alphabet_len), 
            dtype=torch.float32, 
            device=latent.device
        )

        # Start Token
        inp = torch.full(
            fill_value=-32,
            size=(batch_size, 1, self.alphabet_len), 
            dtype=torch.float32, 
            device=latent.device
        )
        inp[:, :, 0] = 32
        y[:, 0, :] = inp.squeeze().detach()

        # Iterative Token Prediction
        for i in range(1, self.smile_len):
            prediction, hidden = self.decode_next(inp, hidden)
            y[:, i, :] = prediction
            
            if target is not None:
                inp = target[:, i, :].unsqueeze(1)
            else:
                inp = prediction.unsqueeze(1).detach()

        return y

import torch
import torch.nn as nn

class DecodeNext(nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        self.gru = nn.GRU(
            input_size=self.params.ALPHABET_LEN, 
            hidden_size=self.params.GRU_HIDDEN_DIM, 
            batch_first=True
        )
        
        self.dense_decoder = nn.Sequential(
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ), 
            nn.Tanh(), 
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.ALPHABET_LEN
            ), 
            nn.Tanh()
        )

    def forward(self, inp, hidden):
        
        out, hidden = self.gru(inp, hidden)
    
        out = self.dense_decoder(out)

        return out, hidden

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        self.decode_next = DecodeNext(params)
    
    def forward(self, hidden, target=None):
        
        *_, batch_size, _ = hidden.shape
        
        y = torch.zeros(
            size=(batch_size, self.params.SMILE_LEN, self.params.ALPHABET_LEN), 
            dtype=torch.float32
        )

        inp = torch.zeros(
            size=(batch_size, 1, self.params.ALPHABET_LEN), 
            dtype=torch.float32
        )
        inp[:, :, self.params.stoi['<BOS>']] = 1
        y[:, 0, :] = inp.squeeze()
        
        for i in range(1, self.params.SMILE_LEN):

            prediction, hidden = self.decode_next(inp, hidden)

            y[:, i, :] = prediction.squeeze()

            if target != None:
                inp = target[:, i:i+1, :]
            else:
                inp = prediction

        return y
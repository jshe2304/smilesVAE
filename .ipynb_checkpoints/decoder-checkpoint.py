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
            
            #nn.Dropout(p=0.1), 
            #nn.BatchNorm1d(num_features=self.params.GRU_HIDDEN_DIM), 
            
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.ALPHABET_LEN
            ), 
            nn.Tanh(), 
            
            #nn.Dropout(p=0.1), 
            #nn.BatchNorm1d(num_features=self.params.ALPHABET_LEN), 
            
            nn.Linear(
                in_features=self.params.ALPHABET_LEN, 
                out_features=self.params.ALPHABET_LEN
            )
        )

    def forward(self, inp, hidden):

        # inp.shape:    (N, L, H_in)
        # hidden.shape: (D * num_layers, N, H_out)
        
        out, hidden = self.gru(inp, hidden)
        # out.shape:    (N, L, D * H_out)
        # hidden.shape: (D * num_layers, N, H_out)
        
        # out.shape:    (N, L, D * H_out) -> (N, D * H_out)
        
        prediction = self.dense_decoder(out.squeeze(1))
        
        # prediction.shape: (N, ALPHABET_LEN)

        return prediction, hidden

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        self.dense_decoder = nn.Sequential(
            nn.Linear(
                in_features=self.params.LATENT_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ),
            nn.Tanh(), 
            
            #nn.Dropout(p=0.1), 
            #nn.BatchNorm1d(num_features=self.params.GRU_HIDDEN_DIM), 
            
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ), 
            nn.Tanh(), 
            
            #nn.Dropout(p=0.1), 
            #nn.BatchNorm1d(num_features=self.params.GRU_HIDDEN_DIM), 
            
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ), 
        )
        
        self.decode_next = DecodeNext(params)
    
    def forward(self, latent, target=None, softmax=True):
        # latent.shape = (N, LATENT_DIM)
        
        hidden = self.dense_decoder(latent).unsqueeze(0)
        # hidden.shape = (1, N, GRU_HIDDEN_DIM)

        *_, batch_size, _ = hidden.shape
        
        y = torch.zeros(
            size=(batch_size, self.params.SMILE_LEN, self.params.ALPHABET_LEN), 
            dtype=torch.float32
        )

        inp = torch.full(
            fill_value=-16,
            size=(batch_size, 1, self.params.ALPHABET_LEN), 
            dtype=torch.float32
        )
        inp[:, :, self.params.stoi['<BOS>']] = 16
        y[:, 0, :] = inp.squeeze()

        for i in range(1, self.params.SMILE_LEN):
            
            # inp.shape = (N, 1, ALPHABET_LEN)
            prediction, hidden = self.decode_next(inp, hidden)
            # prediction.shape: (N, ALPHABET_LEN)

            y[:, i, :] = prediction

            if target != None:
                inp = target[:, i:i+1, :]
            else:
                inp = prediction.unsqueeze(1)

        return y

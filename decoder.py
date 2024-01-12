import torch
import torch.nn as nn

class DecodeNext(nn.Module):
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
                out_features=self.params.GRU_HIDDEN_DIM, 
            ), 
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.ALPHABET_LEN
            )
        )

    def forward(self, inp, hidden):

        out, hidden = self.gru(inp, hidden)

        prediction = self.fc_network(out.squeeze(1))

        return prediction, hidden

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        
        self.fc_network = nn.Sequential(
            nn.Linear(
                in_features=self.params.LATENT_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            ), 
            nn.Linear(
                in_features=self.params.GRU_HIDDEN_DIM, 
                out_features=self.params.GRU_HIDDEN_DIM
            )
        )
        
        self.decode_next = DecodeNext(params)
    
    def forward(self, latent, target=None):
        # latent.shape = (N, LATENT_DIM)
        
        hidden = self.fc_network(latent).unsqueeze(0)
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

import torch
import torch.nn as nn

class DecodeNext(nn.Module):
    '''
    Last Input, Hidden State -> Prediction Logits, Updated Hidden State
    '''
    
    def __init__(self, hidden_size=384, alphabet_len=32, embedding_dim=8, *args, **kwargs):
        super().__init__()

        self.embed = nn.Embedding(
            num_embeddings=alphabet_len, 
            embedding_dim=embedding_dim
        )
        
        self.rnn = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            batch_first=True, 
        )
        
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, alphabet_len), 
            nn.ReLU(), 
            nn.Linear(alphabet_len, alphabet_len)
        )

    def forward(self, inp, hidden):
        inp = self.embed(inp)
        out, hidden = self.rnn(inp, hidden)
        prediction = self.dense(out)
        return prediction, hidden

class Decoder(nn.Module):
    '''
    Iteratively predict token sequence from provided hidden state. 
    
    Hidden  -->  Dense  -->  GRU  -->  Logits Sequence
    '''
    
    def __init__(self, L, hidden_size=256, alphabet_len=32, smile_len=65, embedding_dim=8, *args, **kwargs):
        super().__init__()
        
        self.alphabet_len = alphabet_len
        self.smile_len = smile_len
        
        # Dense Layers
        self.dense = nn.Sequential(
            nn.Linear(L, hidden_size), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(hidden_size, hidden_size)
        )
        
        # GRU Unit
        self.decode_next = DecodeNext(hidden_size, alphabet_len, embedding_dim)
    
    def forward(self, latent, target=None):
        
        # Dense Layers
        batch_size = latent.size(0)
        hidden = self.dense(latent).unsqueeze(0)

        # Outputs Sequence
        outputs = []

        # Input BOS indices
        inp = torch.zeros(
            size=(batch_size, 1), 
            dtype=torch.long, 
            device=latent.device
        )
        
        # Iterative Token Prediction
        for i in range(self.smile_len):
            prediction, hidden = self.decode_next(inp, hidden)
            outputs.append(prediction)

            if target is not None:
                inp = target[:, i].unsqueeze(1)
            else:
                _, top1 = prediction.topk(1)
                inp = top1.squeeze(-1).detach()

        outputs = torch.cat(outputs, dim=1)

        return outputs

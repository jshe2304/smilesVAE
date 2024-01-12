import torch
import torch.nn as nn

def to_one_hot(smiles: list[str], params):
    
    if type(smiles) == str: smiles = [smiles]
    
    one_hots = torch.zeros(
        size=(len(smiles), params.SMILE_LEN, params.ALPHABET_LEN), 
        dtype=torch.float32
    )

    for i, smile in enumerate(smiles):
        
        # start sequence with BOS
        one_hots[i, 0, params.stoi['<BOS>']] = 1
        
        for j, char in enumerate(smile, start=1):
            one_hots[i, j, params.stoi[char]] = 1
        
        # fill rest of sequence with <EOS>
        one_hots[i, (len(smile) + 1):, params.stoi['<EOS>']] = 1
    
    return one_hots

def from_one_hot(one_hot_smiles, params, show_padding=False):
    smiles = []
    
    for one_hot_smile in one_hot_smiles:
        smile = ""
        for one_hot in one_hot_smile:
            c = params.itos[int(torch.argmax(one_hot))]
            if c == '<BOS>': smile += '<' if show_padding else ''
            elif c == '<EOS>': smile += '>' if show_padding else ''
            else: smile += c
        smiles.append(smile)
    
    return smiles

def stos(encoder, decoder, smile, params):
    x = to_one_hot(smile, params)
    
    latent = encoder(x)
    
    y = decoder(latent)
    
    return str(from_one_hot(torch.softmax(y, dim=2), params))

def pack(smiles: list[str], ctoi, embedding_dim=16):
    
    # Create index-wise vectorization
    
    vectorized_smiles = [torch.Tensor([ctoi.get(c) for c in smile]) for smile in smiles]

    smiles_lengths = [len(smile) for smile in vectorized_smiles]
    
    # Create Padded Tensor
    
    max_len = max(len(smile) for smile in smiles)
    
    smiles_tensor = torch.zeros(size=(len(vectorized_smiles), max_len), dtype=torch.int)
    for i, smile in enumerate(vectorized_smiles):
        smiles_tensor[i, :len(smile)] = smile

    # Pack
    
    return nn.utils.rnn.pack_padded_sequence(smiles_tensor, smiles_lengths, batch_first=True)

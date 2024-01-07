import torch

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

def from_one_hot(one_hot_smiles, params):
    smiles = []
    
    for one_hot_smile in one_hot_smiles:
        smile = ""
        for one_hot in one_hot_smile:
            c = params.itos[int(torch.argmax(one_hot))]
            smile += c #if c != '<BOS>' and c != '<EOS>' else ''
        smiles.append(smile)
    
    return smiles
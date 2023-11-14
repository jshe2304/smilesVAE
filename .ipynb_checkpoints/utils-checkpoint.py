import numpy as np
import pandas as pd
import torch
import os
import json
from collections import namedtuple

def fetch_smiles_zinc(zinc_dir: str):
    smiles = []

    for root, dirs, files in os.walk(zinc_dir):
        for file in files:
            if not file.endswith('.smi'):
                continue
            path = os.path.join(root, file)
            smiles.append(pd.read_table(path, sep=' '))

    smiles = pd.concat(smiles)
    
    return smiles

def fetch_smiles_gdb13(zinc_dir: str):
    smiles = []

    for root, dirs, files in os.walk(zinc_dir):
        for file in files:
            if not file.endswith('.smi'):
                continue
            path = os.path.join(root, file)
            smiles.append(pd.read_table(path, header=None))

    smiles = pd.concat(smiles)
    
    return smiles

def to_one_hot(smiles: list[str], params):
    
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
            smile += c if c != '<BOS>' and c != '<EOS>' else ''
        smiles.append(smile)
    
    return smiles

Params = namedtuple(
    'Params', 
    ['N', 'SMILE_LEN', 'alphabet', 'ALPHABET_LEN', 'stoi', 'itos', 'GRU_HIDDEN_DIM', 'LATENT_DIM']
)

def make_params(smiles=None, GRU_HIDDEN_DIM=128, LATENT_DIM=128, from_file=None, to_file=None):
    if smiles is None and not from_file:
        raise ValueError(
            'make_params: cannot create parameters, need smiles or from_file'
        )
    
    if from_file:
        with open(from_file) as f:
            return Params(**json.load(f))
    
    alphabet = set()
    for smile in smiles: alphabet.update(smile)
    alphabet = sorted(list(alphabet))
    alphabet = ['<BOS>'] + alphabet + ['<EOS>']
    
    params = Params(
        N = len(smiles), 
        SMILE_LEN = max(len(smile) for smile in smiles) + 2, 
        alphabet = alphabet, 
        ALPHABET_LEN = len(alphabet), 
        stoi = {c: int(i) for i, c in enumerate(alphabet)},
        itos = alphabet, 
        GRU_HIDDEN_DIM = GRU_HIDDEN_DIM, 
        LATENT_DIM = LATENT_DIM
    )
    
    if to_file:
        with open(to_file, 'w') as f:
            f.write(json.dumps(params._asdict()))
    
    return params
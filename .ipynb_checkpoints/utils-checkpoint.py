import numpy as np
import pandas as pd
import torch
import os
import json
from collections import namedtuple
import random
from embedding_utils import *

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

def fetch_smiles_gdb13(datadir: str):

    train = pd.read_table(os.path.join(datadir, 'train_gdb13.smi'), header=None)[0]
    test = pd.read_table(os.path.join(datadir, 'test_gdb13.smi'), header=None)[0]
    
    return list(train), list(test)

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

def token_accuracy(x, y):
    x = torch.argmax(x, dim=2)
    y = torch.argmax(y, dim=2)
    
    return float(torch.mean(
        torch.mean(
            (x == y), 
            dim=1, 
            dtype=torch.float32
        )
    ))

def percent_reconstructed(x, x_hat):
    x = torch.argmax(x, dim=2)
    x_hat = torch.argmax(x_hat, dim=2)
    
    return float(torch.mean(
        torch.all(x == x_hat, dim=1), 
        dtype=torch.float32
    ))

def evaluate_ae(encoder, decoder, smiles, eval_n, params):
    
    encoder.eval()
    decoder.eval()
    
    smiles = random.sample(smiles, eval_n)
    
    one_hots = to_one_hot(smiles, params)
    
    with torch.no_grad():
        _, _, z = encoder(one_hots)
        x_hat = decoder(z)
    
    pred_smiles = from_one_hot(torch.softmax(x_hat, dim=2), params)
    
    return pd.DataFrame({'target': smiles, 'predicted': pred_smiles})

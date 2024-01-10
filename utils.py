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

def fetch_smiles_gdb13(zinc_dir: str):
    smiles = []

    for root, dirs, files in os.walk(zinc_dir):
        for file in files:
            if not file.endswith('.smi'):
                continue
            path = os.path.join(root, file)
            smiles.append(pd.read_table(path, header=None))

    smiles = pd.concat(smiles)
    
    return smiles[0]


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
        SMILE_LEN = max(len(smile) for smile in smiles), 
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

def mean_similarity(x, y):
    x_argmax = torch.argmax(x, dim=2)
    y_argmax = torch.argmax(y, dim=2)
    
    return float(torch.mean(
        torch.mean(
            (x_argmax == y_argmax), 
            dim=1, 
            dtype=torch.float32
        )
    ))

def evaluate_ae(encoder, decoder, smiles, eval_n, params):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        x = to_one_hot(random.sample(smiles, eval_n), params)

        _, _, z = encoder(x)

        y = decoder(z)

        print(f"mean token matching: {mean_similarity(x, y)}\n")

        sample_smiles = random.sample(smiles, 10)

        one_hots = to_one_hot(sample_smiles, params)

        _, _, z = encoder(one_hots)

        y = decoder(z)

        out_smiles = from_one_hot(torch.softmax(y, dim=2), params)

        print(sample_smiles)
        print(out_smiles)
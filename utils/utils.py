import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import os
import json
from collections import namedtuple
import random

# =============
# Preprocessing
# =============

def make_dataspec(DATADIR):
    '''
    Make alphabet, stoi tables from data files
    Stores in a json info file
    '''
    
    train = list(pd.read_csv(os.path.join(DATADIR, 'train.smi'))['smiles'])
    test = list(pd.read_csv(os.path.join(DATADIR, 'test.smi'))['smiles'])
    total = train + test
    
    smile_len = 2 + max(len(smile) for smile in total)
    
    alphabet = set()
    for smile in total: 
        alphabet.update(smile)
    alphabet = sorted(list(alphabet))
    alphabet = ['<BOS>'] + alphabet + ['<EOS>']
    
    stoi = {c: int(i) for i, c in enumerate(alphabet)}
    
    with open(os.path.join(DATADIR, 'spec.json'), 'w') as f:
        f.write(json.dumps({
            'n': len(total), 
            'n_train': len(train), 
            'n_test': len(test), 
            'smile_len': smile_len, 
            'alphabet': alphabet, 
            'stoi': stoi
        }, indent=4))

def fetch_params(FNAME):
    '''
    Make a params object from json file
    '''
    with open(FNAME) as f:
        params_dict = json.load(f)
    
    Params = namedtuple(
        'Params', 
        params_dict.keys()
    )
    
    return Params(**params_dict)

def make_params(FNAME=None, **kwargs):
    '''
    Make a json file or params object from params
    '''
    if FNAME:
        with open(FNAME, 'w') as f:
            f.write(json.dumps(kwargs, indent=4))
            
    Params = namedtuple(
        'Params', 
        kwargs.keys()
    )
    
    return Params(**kwargs)

# =========
# Embedding
# =========

def to_hot(smiles: list[str], stoi, smile_len):
    
    if type(smiles) == str: smiles = [smiles]
    
    one_hots = torch.zeros(
        size=(len(smiles), smile_len, len(stoi)), 
        dtype=torch.float32
    )

    for i, smile in enumerate(smiles):
        
        one_hots[i, 0, stoi['<BOS>']] = 1
        
        for j, char in enumerate(smile, start=1):
            one_hots[i, j, stoi[char]] = 1
        
        one_hots[i, (len(smile) + 1):, stoi['<EOS>']] = 1
    
    return one_hots

def from_hot(one_hot_smiles, alphabet, show_padding=False):
    smiles = []
    
    for one_hot_smile in one_hot_smiles:
        smile = ""
        for one_hot in one_hot_smile:
            c = alphabet[int(torch.argmax(one_hot))]
            if c == '<BOS>': smile += '<' if show_padding else ''
            elif c == '<EOS>': smile += '>' if show_padding else ''
            elif c == 'L': smile += 'Cl'
            else: smile += c
        smiles.append(smile)
    
    return smiles

# ==============
# Evaluate Model
# ==============

def token_accuracy(x, x_hat):
    x = torch.argmax(x, dim=2)
    x_hat = torch.argmax(x_hat, dim=2)
    
    return float(torch.mean(
        torch.mean(
            (x == x_hat), 
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

# ===============
# Latent Analysis
# ===============

def get_latent_distributions(encoder, x, n=1000):
    latents, _, _ = encoder(x)
    
    means = torch.mean(latents, dim=0)
    stds = torch.std(latents, dim=0)
    
    return means, stds

def get_important_dimensions(x, encoder, decoder=None, n=1000, use_loss=False):
    
    # Calculate based on loss impact
    if use_loss:
        CE_LOSS = lambda expect, predict: float(torch.nn.functional.cross_entropy(
            predict.transpose(1, 2), 
            expect.argmax(dim=2)
        ))

        z, _, _ = encoder(x)
        
        L = z.size(1)

        reference_loss = CE_LOSS(x, decoder(z))

        losses = [0] * L

        for i in range(L):
            dim_values = z[:, i].clone()
            z[:, i] = 0 # Zero the dimension
            losses[i] = CE_LOSS(x, decoder(z)) - reference_loss
            z[:, i] = dim_values
            
        return [i for i, loss in enumerate(losses) if loss > 0.01]
    
    # Calculate based on value standard deviation
    else:
        _, stds = get_latent_distributions(encoder, x, n=n)
        
        return [i for i, std in enumerate(stds) if std > 0.05]
        


# ============
# Optimization
# ============

def decompress(compressed_x, uncompressed_dims, n, noised=False):
    if compressed_x.size(-1) >= n: return compressed_x
    
    if compressed_x.dim() > 1:
        x = torch.zeros(compressed_x.size(0), n, dtype=compressed_x.dtype)
        
        for i, dim in enumerate(uncompressed_dims):
            x[:, dim] = compressed_x[:, i]
    else:
        x = torch.zeros(n, dtype=compressed_x.dtype)
        
        for i, dim in enumerate(uncompressed_dims):
            x[dim] = compressed_x[i]

    return x
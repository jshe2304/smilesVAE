import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd

import time
import os
import random

from utils import *
from encoder import Encoder
from decoder import DecodeNext, Decoder

# Model Parameters

params = make_params(smiles=smiles, GRU_HIDDEN_DIM=256, LATENT_DIM=128)

# Training Parameters

n = 1000
batch_size = 32
LR = 0.0001
EPOCHS = 1

# Training Data

smiles = list(fetch_smiles_gdb13('./data/gdb13/')[0])

one_hots = to_one_hot(random.sample(smiles, n), params)

train_dataloader = DataLoader(one_hots, batch_size=batch_size, shuffle=True)

# Model

encoder = Encoder(params)
decoder = Decoder(params)

encoder.train()
decoder.train()

# Optimizer and Losses

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

CE_loss = nn.CrossEntropyLoss()
KL_divergence = lambda z_mean, z_logvar : -0.5 * torch.sum(1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar))

# Training Loop

log_file = 'log.csv'
i = 0

for epoch_n in range(EPOCHS):
    for x in train_dataloader:

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # VAE Forward
        
        z_mean, z_logvar, z = encoder(x)
        y = decoder(z, target=x)
        
        # Loss
        
        loss = CE_loss(y.transpose(1, 2), torch.argmax(x, dim=2)) + \
               KL_divergence(z_mean, z_logvar) * 0.01
        
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        # Log
        
        if (i % 10) == 0:
            encoder.eval()
            decoder.eval()
            
            # Evaluate sample
            
            with torch.no_grad():
                x = to_one_hot(random.sample(smiles, 100), params)
                _, _, z = encoder(x)
                y = decoder(z)

            similarity = mean_similarity(x, y)
            
            # Output to log
            
            with open(log_file, "a") as f:
                f.write(f'{i}, {float(loss)}, {similarity}\n')
            
            encoder.train()
            decoder.train()
            
            # Save parameters
            
            if (i % 20) == 0:
                torch.save(encoder.state_dict(), 'encoder_weights.pth')
                torch.save(decoder.state_dict(), 'decoder_weights.pth')
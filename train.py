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
from embedding_utils import *
from encoder import Encoder
from decoder import DecodeNext, Decoder

print(f"Imports done...")

# Training Parameters

n = 1500000
batch_size = 16
LR = 0.0001
EPOCHS = 5

# Training Data

print(f"Loading training data...")

smiles = list(fetch_smiles_gdb13('./data/gdb13/')[0])

params = make_params(smiles=smiles, GRU_HIDDEN_DIM=256, LATENT_DIM=128)

one_hots = to_one_hot(random.sample(smiles, n), params)

train_dataloader = DataLoader(one_hots, batch_size=batch_size, shuffle=True)

print("Done.")

# Model

print("Training Model...")

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

log_file = 'train_output/log.csv'

with open(log_file, "w") as f:
    f.write("i,loss,similarity\n")

i = 0

start_time = time.time()

for epoch_n in range(EPOCHS):
    print(f"Epoch {epoch_n}...")

    for x in train_dataloader:

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # VAE Forward
        
        z_mean, z_logvar, z = encoder(x)
        x_hat = decoder(z, target=x)
        
        # Loss
        
        #loss = CE_loss(y.transpose(1, 2), torch.argmax(x, dim=2)) + KL_divergence(z_mean, z_logvar) * 0.01
        
        loss = CE_loss(x_hat.transpose(1, 2), torch.argmax(x, dim=2))
        
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
                x_hat = decoder(z)

            similarity = mean_similarity(x, x_hat)
            
            # Output to log
            
            with open(log_file, "a") as f:
                f.write(f'{i},{float(loss)},{similarity}\n')
            
            encoder.train()
            decoder.train()
            
            # Save parameters
            
            if (i % 20) == 0:
                torch.save(encoder.state_dict(), 'weights/encoder_weights.pth')
                torch.save(decoder.state_dict(), 'weights/decoder_weights.pth')
        
        i += 1

print(f"Done. Time Elapsed: {time.time() - start_time}")

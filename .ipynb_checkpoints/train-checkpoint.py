import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import os
import random
import math

from utils import *
from embedding_utils import *
from encoder import Encoder
from decoder import Decoder

print(f"Imports done...")

# Training Parameters

n = 1600000
batch_size = 16
LR = 0.00000001
EPOCHS = 10
from_existing = False
CE_label_smoothing = 0
data_dir = './data/gdb13/'
weights_dir = './weights-3/'
log_file = './logs/log-3.csv'

# Training Data

print(f"Loading training data...")

smiles = fetch_smiles_gdb13(data_dir).values.tolist()

for i, smile in enumerate(smiles):
    smiles[i] = smile.replace('Cl', 'L')

params = make_params(smiles=smiles, GRU_HIDDEN_DIM=512, LATENT_DIM=256)

one_hots = to_one_hot(random.sample(smiles, n), params)

train_dataloader = DataLoader(one_hots, batch_size=batch_size, shuffle=True)

print("Done.")

# Model

print("Training Model...")

encoder = Encoder(params)
decoder = Decoder(params)

if from_existing:
    encoder.load_state_dict(torch.load(weights_dir + 'encoder_weights.pth'))
    decoder.load_state_dict(torch.load(weights_dir + 'decoder_weights.pth'))

encoder.train()
decoder.train()

# Optimizer and Losses

encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

#class_weights = torch.full((params.ALPHABET_LEN, ), 1, dtype=torch.float32)
#class_weights[params.stoi['C']] = 0.3

CE_loss = nn.CrossEntropyLoss(label_smoothing=CE_label_smoothing)
KL_divergence = lambda z_mean, z_logvar : -0.5 * torch.mean(1 + z_logvar - z_mean ** 2 - torch.exp(z_logvar))
logistic = lambda x: 1/(1 + math.exp(-x))

# Training Loop

if not from_existing:
    with open(log_file, "w") as f:
        f.write("i,time,loss,similarity\n")

i = 0

start_time = time.time()

for epoch_n in range(EPOCHS):
    print(f"Epoch {epoch_n}...")

    #KL_weight = 0.1 * logistic(epoch_n - EPOCHS * 0.6)
    
    for x in train_dataloader:

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        # VAE Forward
        
        z_mean, z_logvar, z = encoder(x)
        x_hat = decoder(z)
        
        x_hat = x_hat.transpose(1, 2)
        x_labels = torch.argmax(x, dim=2)

        # Loss

        loss = CE_loss(x_hat, x_labels) #+ KL_divergence(z_mean, z_logvar) * KL_weight
        loss.backward()
        
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        # Logging

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
                f.write(f'{i},{time.time() - start_time},{float(loss)},{similarity}\n')
            
            encoder.train()
            decoder.train()
            
            # Save parameters
            
            if (i % 20) == 0:
                torch.save(encoder.state_dict(), weights_dir + 'encoder_weights.pth')
                torch.save(decoder.state_dict(), weights_dir + 'decoder_weights.pth')
        
        i += 1

print(f"Done. Time Elapsed: {time.time() - start_time}")
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

# ===================
# TRAINING PARAMETERS
# ===================

BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 20
KL_WEIGHT = 1
KL_ANNEAL = None

NEW_RUN = False
DATADIR = './data/gdb13/'
OUTDIR = './run-6/'
LOGFILE = os.path.join(OUTDIR, 'log.csv')

CE_LOSS = nn.CrossEntropyLoss()
KL_DIV = lambda mean, logvar : -0.5 * torch.mean(1 + logvar - mean ** 2 - torch.exp(logvar))
LOGISTIC = lambda x: 1/(1 + math.exp(-x))

if __name__ == "__main__":
    
    # ==========
    # DATALOADER
    # ==========
    
    smiles_train, smiles_test = fetch_smiles_gdb13(DATADIR)
    
    params = make_params(smiles=smiles_train + smiles_test, GRU_HIDDEN_DIM=512, LATENT_DIM=256)
    
    smiles_tensor = to_one_hot(smiles_train, params)
    
    train_dataloader = DataLoader(smiles_tensor, batch_size=BATCH_SIZE, shuffle=True)
    
    # =================
    # MODEL & OPTIMIZER
    # =================
    
    encoder = Encoder(params)
    decoder = Decoder(params)

    if not NEW_RUN:
        encoder.load_state_dict(torch.load(OUTDIR + 'encoder_weights.pth'))
        decoder.load_state_dict(torch.load(OUTDIR + 'decoder_weights.pth'))

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)

    # =============
    # TRAINING LOOP
    # =============
    
    i = 0
    
    start_time = time.time()

    if NEW_RUN or not os.path.isfile(LOGFILE):
        with open(LOGFILE, "w") as f:
            f.write("i,time,loss,ce,kl,accuracy,prec\n")
    else:
        # https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
        with open(LOGFILE, 'rb') as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
                
                last_line = f.readline().decode().split(',')
                i = int(last_line[0])
                start_time = float(last_line[1])
            except OSError:
                pass

    for epoch in range(EPOCHS):

        kl_weight = KL_WEIGHT * LOGISTIC(epoch - EPOCHS * KL_ANNEAL) if KL_ANNEAL else 1

        for x in train_dataloader:
            
            encoder.train()
            decoder.train()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            # ============
            # FORWARD PASS
            # ============

            mean, logvar, z = encoder(x)
            x_hat = decoder(z)

            x_hat = x_hat.transpose(1, 2)
            x_labels = torch.argmax(x, dim=2)
            
            # ============
            # Compute Loss
            # ============

            ce = CE_LOSS(x_hat, x_labels)
            kl = KL_DIV(mean, logvar)
            
            loss = ce + kl * kl_weight
            
            # ====================
            # BACKWARD PROPAGATION
            # ====================
            
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            # ===========
            # Log Metrics
            # ===========

            if (i % 100) == 0:
                encoder.eval()
                decoder.eval()
                
                # ======================
                # Test Sample Evaluation
                # ======================

                with torch.no_grad():
                    x = to_one_hot(random.sample(smiles_test, 100), params)
                    _, _, z = encoder(x)
                    x_hat = decoder(z)

                acc = token_accuracy(x, x_hat)
                
                prec = percent_reconstructed(x, x_hat)
                
                # ============
                # Write to Log
                # ============

                with open(LOGFILE, "a") as f:
                    t = time.time() - start_time
                    f.write(
                        f'{i},{t},{float(loss)},{float(ce)},{float(kl)},{acc},{prec}\n'
                    )
            
            # ==========
            # Save Model
            # ==========

            if (i % 200) == 0:
                torch.save(encoder.state_dict(), OUTDIR + 'encoder_weights.pth')
                torch.save(decoder.state_dict(), OUTDIR + 'decoder_weights.pth')

            i += 1

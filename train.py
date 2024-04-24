import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import json
import os
import random
import math
import sys

from utils.utils import *
from utils.dataset import *
from rnn_vae.encoder import Encoder
from rnn_vae.decoder import Decoder
from rnn_vae.predictor import Predictor

# ==========
# FILE PATHS
# ==========

DATADIR = './data/gdb13/'
OUTDIR = sys.argv[1]
assert os.path.isdir(OUTDIR)

DATASPEC_FILE = os.path.join(DATADIR, 'spec.json')
LOG_FILE = os.path.join(OUTDIR, 'log.csv')
RUNSPEC_FILE = os.path.join(OUTDIR, 'runspec.json')

ENCODER_WEIGHTS_FILE = os.path.join(OUTDIR, 'encoder_weights.pth')
DECODER_WEIGHTS_FILE = os.path.join(OUTDIR, 'decoder_weights.pth')
PREDICTOR_WEIGHTS_FILE = os.path.join(OUTDIR, 'predictor_weights.pth')

# ================
# TRAIN PARAMETERS
# ================

with open(RUNSPEC_FILE) as f:
    (L, 
    EPOCHS, BATCH_SIZE, LR, 
    KL_WEIGHT, KL_ANNEAL_AT, KL_ANNEAL_RATE, 
    PRED_WEIGHT, PRED_ANNEAL_AT, PRED_ANNEAL_RATE) = json.load(f).values()
    
print(f'{L}-dimension VAE')
print(f'{EPOCHS} epochs of {BATCH_SIZE} samples')
print(f'Learning at a rate of {LR}')
print(f'Annealing in KL at {KL_ANNEAL_AT} epochs at a rate of {KL_ANNEAL_RATE} with strength {KL_WEIGHT}')
print(f'Annealing in KL at {PRED_ANNEAL_AT} epochs at a rate of {PRED_ANNEAL_RATE} with strength {PRED_WEIGHT}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CE_LOSS = nn.CrossEntropyLoss()
KL_DIV = lambda mean, logvar : -0.5 * torch.mean(1 + logvar - mean ** 2 - torch.exp(logvar))
LOGISTIC = lambda x: 1/(1 + math.exp(-x))

if __name__ == "__main__":

    # =========
    # LOAD DATA
    # =========
    
    dataspec = fetch_params(DATASPEC_FILE)

    trainset, testset = make_data(DATADIR, device)

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # =================
    # MODEL & OPTIMIZER
    # =================
    
    encoder = Encoder(L)
    decoder = Decoder(L)
    predictor = Predictor(L)
    
    if os.path.isfile(ENCODER_WEIGHTS_FILE) and os.path.isfile(DECODER_WEIGHTS_FILE) and os.path.isfile(PREDICTOR_WEIGHTS_FILE):
        encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_FILE))
        decoder.load_state_dict(torch.load(DECODER_WEIGHTS_FILE))
        predictor.load_state_dict(torch.load(PREDICTOR_WEIGHTS_FILE))
    
    encoder.to(device)
    decoder.to(device)
    predictor.to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=LR)

    # =============
    # TRAINING LOOP
    # =============

    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("loss,ce,kl,logp,qed,sas,accuracy,prec\n")
            EPOCHS_COMPLETED = 0
    else:
        with open(LOG_FILE) as f:
            EPOCHS_COMPLETED = sum(1 for _ in f) - 1
    
    for epoch in range(EPOCHS_COMPLETED + 1, EPOCHS_COMPLETED + EPOCHS + 1):
        
        kl_weight = KL_WEIGHT * LOGISTIC(  KL_ANNEAL_RATE * (epoch - KL_ANNEAL_AT)  )
        pred_weight = PRED_WEIGHT * LOGISTIC(  PRED_ANNEAL_RATE * (epoch - PRED_ANNEAL_AT)  )
        
        #print(f'Epoch {epoch}: KL ({kl_weight})\tPred({pred_weight})')

        for x, logp, qed, sas in dataloader:

            encoder.train()
            decoder.train()
            predictor.train()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            predictor_optimizer.zero_grad()
            
            # ============
            # FORWARD PASS
            # ============

            mean, logvar, z = encoder(x)
            x_hat = decoder(z, target=x)
            logp_hat, qed_hat, sas_hat = predictor(z)

            x_hat = x_hat.transpose(1, 2)
            x_labels = torch.argmax(x, dim=2)
            
            # ==============
            # Compute Losses
            # ==============

            ce = CE_LOSS(x_hat, x_labels)
            kl = KL_DIV(mean, logvar)
            
            logp_err = torch.mean((logp_hat - logp) ** 2)
            qed_err = torch.mean((qed_hat - qed) ** 2)
            sas_err = torch.mean((sas_hat - sas) ** 2)
            
            loss = ce + kl * kl_weight + (logp_err + qed_err + sas_err) * pred_weight
            
            # ====================
            # BACKWARD PROPAGATION
            # ====================
            
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            predictor_optimizer.step()
            
        # ===========
        # Log Metrics
        # ===========

        encoder.eval()
        decoder.eval()
        predictor.eval()

        # ======================
        # Test Sample Evaluation
        # ======================

        with torch.no_grad():
            x, _, _, _ = testset.sample(64)
            mean, _, _ = encoder(x)
            x_hat = decoder(mean)

        acc = token_accuracy(x, x_hat)
        prec = percent_reconstructed(x, x_hat)

        # ============
        # Write to Log
        # ============

        with open(LOG_FILE, "a") as f:
            f.write(
                f'{float(loss)},{float(ce)},{float(kl)},{float(logp_err)},{float(qed_err)},{float(sas_err)},{acc},{prec}\n'
            )

        torch.save(encoder.state_dict(), ENCODER_WEIGHTS_FILE)
        torch.save(decoder.state_dict(), DECODER_WEIGHTS_FILE)
        torch.save(predictor.state_dict(), PREDICTOR_WEIGHTS_FILE)
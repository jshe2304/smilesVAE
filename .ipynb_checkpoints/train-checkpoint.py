import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import os
import random
import math
import sys

from utils import *
from dataset import *
from encoder import Encoder
from decoder import Decoder
from predictor import Predictor

# ==========
# FILE PATHS
# ==========

DATADIR = './data/gdb13/'
OUTDIR = sys.argv[1]

DATASPEC_FILE = os.path.join(DATADIR, 'spec.json')
MODELSPEC_FILE = os.path.join(OUTDIR, 'modelspec.json')

ENCODER_WEIGHTS_FILE = os.path.join(OUTDIR, 'encoder_weights.pth')
DECODER_WEIGHTS_FILE = os.path.join(OUTDIR, 'decoder_weights.pth')
PREDICTOR_WEIGHTS_FILE = os.path.join(OUTDIR, 'predictor_weights.pth')
LOG_FILE = os.path.join(OUTDIR, 'log.csv')

# ================
# TRAIN PARAMETERS
# ================

EPOCHS = 32
BATCH_SIZE = 32
LR = 0.00001

KL_WEIGHT = 0.75
KL_ANNEAL = -0.25
KL_ANNEAL_RATE = 0.2

PRED_WEIGHT = 0.3
PRED_ANNEAL = -0.25
PRED_ANNEAL_RATE = 0.2

modelspec = make_params(hidden_dim=512, latent_dim=256, pred_dim=64, dropout=0.15)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CE_LOSS = nn.CrossEntropyLoss()
KL_DIV = lambda mean, logvar : -0.5 * torch.mean(1 + logvar - mean ** 2 - torch.exp(logvar))
LOGISTIC = lambda x: 1/(1 + math.exp(-x))

if __name__ == "__main__":
    
    if not os.path.isdir(OUTDIR): os.mkdir(OUTDIR)

    # =========
    # LOAD DATA
    # =========
    
    dataspec = fetch_params(DATASPEC_FILE)

    trainset, testset = make_data(DATADIR)

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # =================
    # MODEL & OPTIMIZER
    # =================
    
    encoder = Encoder(modelspec, dataspec)
    decoder = Decoder(modelspec, dataspec)
    predictor = Predictor(modelspec)
    
    if os.path.isfile(ENCODER_WEIGHTS_FILE) and os.path.isfile(DECODER_WEIGHTS_FILE) and os.path.isfile(PREDICTOR_WEIGHTS_FILE):
        encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_FILE))
        decoder.load_state_dict(torch.load(DECODER_WEIGHTS_FILE))
        predictor.load_state_dict(torch.load(PREDICTOR_WEIGHTS_FILE))

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=LR)

    # =============
    # TRAINING LOOP
    # =============
    
    i = 0
    
    start_time = time.time()

    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("i,time,loss,ce,kl,logp,qed,sas,accuracy,prec\n")
    else:
        with open(LOG_FILE, 'rb') as f:
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

        kl_weight = KL_WEIGHT #* LOGISTIC(KL_ANNEAL_RATE * (epoch - EPOCHS * KL_ANNEAL)) #if KL_ANNEAL else 1
        pred_weight = PRED_WEIGHT #* LOGISTIC(KL_ANNEAL_RATE * (epoch - EPOCHS * PRED_ANNEAL)) #if PRED_ANNEAL else 1

        for x, logp, qed, sas in dataloader:
            x.to(device)

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
            logp_hat, qed_hat, sas_hat = predictor(z)
            x_hat = decoder(z)

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

            if (i % 100) == 0:
                encoder.eval()
                decoder.eval()
                predictor.eval()
                
                # ======================
                # Test Sample Evaluation
                # ======================

                with torch.no_grad():
                    x = testset.hots[torch.randint(0, dataspec.n_test, (64,))]
                    mean, _, _ = encoder(x)
                    x_hat = decoder(mean)

                acc = token_accuracy(x, x_hat)
                prec = percent_reconstructed(x, x_hat)
                
                # ============
                # Write to Log
                # ============
                
                with open(LOG_FILE, "a") as f:
                    t = time.time() - start_time
                    f.write(
                        f'{i},{t},\
                        {float(loss)},{float(ce)},{float(kl)},\
                        {float(logp_err)},{float(qed_err)},{float(sas_err)},\
                        {acc},{prec}\n'
                    )
            
            # ==========
            # Save Model
            # ==========

            if (i % 200) == 0:
                torch.save(encoder.state_dict(), ENCODER_WEIGHTS_FILE)
                torch.save(decoder.state_dict(), DECODER_WEIGHTS_FILE)
                torch.save(predictor.state_dict(), PREDICTOR_WEIGHTS_FILE)

            i += 1

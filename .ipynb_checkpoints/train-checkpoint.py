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
from models.MixedNN import Encoder, Decoder, Predictor

# ==========
# FILE PATHS
# ==========

DATADIR = './data/gdb13-augmented/'
OUTDIR = sys.argv[1]
assert os.path.isdir(OUTDIR)

DATASPEC_FILE = os.path.join(DATADIR, 'spec.json')
LOG_FILE = os.path.join(OUTDIR, 'log.csv')
RUNSPEC_FILE = os.path.join(OUTDIR, 'spec.json')

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
    PRED_WEIGHT, PRED_ANNEAL_AT, PRED_ANNEAL_RATE, 
    FORCING_ANNEAL_AT, FORCING_ANNEAL_RATE) = json.load(f).values()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

CrossEntropy = nn.CrossEntropyLoss()
MeanSquaredError = nn.MSELoss()
KLDivergence = lambda mean, logvar : -0.5 * torch.mean(1 + logvar - mean ** 2 - torch.exp(logvar))
Sigmoid = lambda x: 1/(1 + math.exp(-x))

if __name__ == "__main__":

    # =========
    # LOAD DATA
    # =========
    
    dataspec = fetch_params(DATASPEC_FILE)

    to_indices, from_distribution = make_embed_utils(dataspec)

    trainset, testset = make_data(DATADIR, to_indices, device)

    dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    # =================
    # MODEL & OPTIMIZER
    # =================

    kwargs = {
        'L': L, 
        'alphabet_len': len(dataspec.alphabet), 
        'smile_len': dataspec.smile_len
    }
    
    encoder = Encoder(**kwargs)
    decoder = Decoder(**kwargs)
    predictor = Predictor(**kwargs)
    
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
        
        kl_weight = KL_WEIGHT * Sigmoid(  KL_ANNEAL_RATE * (epoch - KL_ANNEAL_AT)  )
        pred_weight = PRED_WEIGHT * Sigmoid(  PRED_ANNEAL_RATE * (epoch - PRED_ANNEAL_AT)  )
        forcing_rate = Sigmoid( FORCING_ANNEAL_RATE * (epoch - FORCING_ANNEAL_AT) )
        
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
            target = x if (random.random() < forcing_rate) else None
            x_hat = decoder(z, target=target).transpose(1, 2)
            logp_hat, qed_hat, sas_hat = predictor(z)

            # ==============
            # Compute Losses
            # ==============

            ce_err = CrossEntropy(x_hat, x)
            kl_err = KLDivergence(mean, logvar)
            
            logp_err = MeanSquaredError(logp_hat, logp)
            qed_err = MeanSquaredError(qed_hat, qed)
            sas_err = MeanSquaredError(sas_hat, sas)
            
            loss = ce_err + kl_err * kl_weight + (logp_err/100 + qed_err + sas_err/100) * pred_weight
            
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
            x, logp, qed, sas = testset.sample(64)
            
            mean, logvar, z = encoder(x)
            x_hat = decoder(z)
            logp_hat, qed_hat, sas_hat = predictor(z)

            ce_err = CrossEntropy(x_hat.transpose(1, 2), x)
            kl_err = KLDivergence(mean, logvar)
            logp_err = MeanSquaredError(logp_hat, logp)
            qed_err = MeanSquaredError(qed_hat, qed)
            sas_err = MeanSquaredError(sas_hat, sas)
            loss = ce_err + kl_err * kl_weight + (logp_err/100 + qed_err + sas_err/100) * pred_weight

            x_hat = x_hat.argmax(dim=2)
            acc = float(torch.mean(torch.mean(x == x_hat, dim=1, dtype=torch.float16)))
            prec = float(torch.mean(torch.all(x == x_hat, dim=1), dtype=torch.float16))

        # ============
        # Write to Log
        # ============
        
        metrics = (loss, ce_err, kl_err, logp_err, qed_err, sas_err, acc, prec)
        with open(LOG_FILE, "a") as f:
            f.write(','.join(str(float(metric)) for metric in metrics) + '\n')

        torch.save(encoder.state_dict(), ENCODER_WEIGHTS_FILE)
        torch.save(decoder.state_dict(), DECODER_WEIGHTS_FILE)
        torch.save(predictor.state_dict(), PREDICTOR_WEIGHTS_FILE)
